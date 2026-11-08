// Tokenizer.java
package RN.transformer;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import RN.utils.NDArrayUtils;

public class Tokenizer implements Serializable {
    private static final long serialVersionUID = -1691008595018974489L;
    
    // Utiliser LinkedHashMap pour préserver l'ordre d'insertion
    private final LinkedHashMap<String, Integer> tokenToId;
    private final LinkedHashMap<Integer, String> idToToken;
    private int vocabSize;
    public INDArray pretrainedEmbeddings;
    private final int embeddingSize;
    private final int maxSequenceLength;


    // Tokens spéciaux
    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final String START_TOKEN = "<START>";
    public static final String END_TOKEN = "<END>";
    
    // WordVectors membre
    private WordVectors wordVectors;

    private List<Integer> lastForwardTokenIds = new ArrayList<>();
    private INDArray embeddingGradients = null;
    
    private Map<Integer, Integer> tokenFrequencies = new HashMap<>();
    
    // Constructeur utilisant des WordVectors
    public Tokenizer(WordVectors wordVectors, int embeddingSize, int maxSequenceLength) {
        this.embeddingSize = embeddingSize;
        this.maxSequenceLength = maxSequenceLength;
        this.tokenToId = new LinkedHashMap<>();
        this.idToToken = new LinkedHashMap<>();
        this.wordVectors = wordVectors;
        
        // Initialiser les tokens spéciaux
        initializeSpecialTokens();
        
        // Ajouter tous les mots du vocabulaire Word2Vec
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            addToken(word); // Maintenant, addToken ne modifie plus pretrainedEmbeddings
        }
        
        this.vocabSize = tokenToId.size();
        
        // Initialiser les embeddings avec WordVectors
        initializeEmbeddings(wordVectors);

        this.embeddingGradients = Nd4j.zeros(vocabSize, embeddingSize);
    }
    
    // Constructeur utilisant une liste de mots
    public Tokenizer(List<String> words, int embeddingSize, int maxSequenceLength) {
        this.embeddingSize = embeddingSize;
        this.maxSequenceLength = maxSequenceLength;
        this.tokenToId = new LinkedHashMap<>();
        this.idToToken = new LinkedHashMap<>();
        this.wordVectors = null; // Pas de WordVectors fourni
        
        // Initialiser les tokens spéciaux
        initializeSpecialTokens();
        
        // Ajouter tous les mots du vocabulaire fourni
        for (String word : words) {
            addToken(word);
        }
        
        this.vocabSize = tokenToId.size();
        
        // Initialiser les embeddings avec des vecteurs aléatoires
        initializeEmbeddings();

        this.embeddingGradients = Nd4j.zeros(vocabSize, embeddingSize);

    }

    /**
     * Convertit une liste de tokens en INDArray d'IDs.
     */
    public INDArray tokensToINDArray(List<String> tokens) {
        int[] ids = tokens.stream()
                          .mapToInt(token -> {
                              int id = tokenToId.getOrDefault(token, getUnkTokenId());
                              updateFrequency(id);
                              return id;
                          })
                          .toArray();
        // Assurer la forme [1, N]
        INDArray arr = Nd4j.createFromArray(ids).castTo(DataType.INT32).reshape(1, ids.length);
        return arr;
    }

    /**
     * Recherche les embeddings pour un batch de séquences d'IDs de tokens.
     *
     * @param tokenIdsBatch INDArray contenant les IDs des tokens du batch [batchSize, seqLength].
     * @return Embeddings du batch [batchSize, seqLength, dModel].
     */
    public INDArray lookupEmbeddings(INDArray tokenIdsBatch) {
        // Vérifier que tokenIdsBatch est de type entier
        if (!tokenIdsBatch.dataType().isIntType()) {
            throw new IllegalArgumentException("tokenIdsBatch doit être de type entier.");
        }

        // tokenIdsBatch shape: [batchSize, seqLength]
        int batchSize = (int) tokenIdsBatch.shape()[0];
        int seqLength = (int) tokenIdsBatch.shape()[1];
        
        // Aplatir les IDs de tokens pour récupérer les embeddings en une seule opération
        INDArray flattenedTokenIds = tokenIdsBatch.reshape(batchSize * seqLength); // [batchSize * seqLength]
        
        // Convertir flattenedTokenIds en int[]
        int[] tokenIds = flattenedTokenIds.toIntVector();
        
        // Stocker les IDs pour le backward pass
        lastForwardTokenIds = Arrays.stream(tokenIds).boxed().collect(Collectors.toList());
        
        // Récupérer les embeddings: [batchSize * seqLength, dModel]
        INDArray batchEmbeddings = getPretrainedEmbeddings().getRows(tokenIds);

        // Mise à l'échelle "Attention Is All You Need" §3.4 : les embeddings
        // appris (word2vec ~1e-3) sont multipliés par sqrt(dModel) avant
        // addition du positional encoding, sinon le signal de contenu est
        // noyé et l'attention reste uniforme (modèle sourd à l'amorce).
        batchEmbeddings = batchEmbeddings.mul(getEmbeddingScale());
        
        // Reshaper en [batchSize, seqLength, dModel]
        batchEmbeddings = batchEmbeddings.reshape(batchSize, seqLength, embeddingSize);

        // Compter les occurrences de 'chat'
        // long chatCount = lastForwardTokenIds.stream().filter(id -> id == tokenToId("chat")).count();
        // System.out.println("Occurrences de 'chat' dans ce batch: " + chatCount);

        return batchEmbeddings;
    }

    /**
     * Facteur d'échelle des embeddings ("Attention Is All You Need" §3.4).
     * Appliqué dans {@link #lookupEmbeddings} et symétriquement dans
     * {@link #backward} via la dérivation en chaîne.
     */
    public double getEmbeddingScale() {
        return Math.sqrt(embeddingSize);
    }

    public Integer tokenToId(String token){
        return getTokenToId().get(token);
    }

    /**
     * Effectue la rétropropagation des gradients à travers les embeddings.
     *
     * @param gradOutput Les gradients de la perte par rapport aux embeddings [batchSize, seqLength, dModel].
     */
    /**
     * Rétropropagation vers les embeddings : accumule, pour chaque position,
     * le gradient sur la ligne d'embedding du token présent à cette position.
     * Les IDs de tokens doivent être fournis explicitement (un même tokenizer
     * sert l'encodeur ET le décodeur, avec des séquences différentes : un
     * cache partagé "derniers IDs" attribuerait les gradients aux mauvais tokens).
     */
    public void backward(INDArray gradOutput, INDArray tokenIdsBatch) {
        // Vérifier la forme de gradOutput
        if (gradOutput.rank() != 3 || gradOutput.size(2) != embeddingSize) {
            throw new IllegalArgumentException("gradOutput doit être de forme [batchSize, seqLength, dModel].");
        }
        if (tokenIdsBatch.rank() != 2 || tokenIdsBatch.size(0) != gradOutput.size(0)
                || tokenIdsBatch.size(1) != gradOutput.size(1)) {
            throw new IllegalArgumentException("tokenIdsBatch doit être de forme [batchSize, seqLength] assortie à gradOutput.");
        }

        int batchSize = (int) gradOutput.size(0);
        int seqLength = (int) gradOutput.size(1);
        int[] tokenIds = tokenIdsBatch.reshape(batchSize * seqLength).toIntVector();

        // Itérer sur chaque élément du batch et de la séquence
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = tokenIds[b * seqLength + s];

                // Récupérer le gradient pour ce token. lookupEmbeddings() multiplie
                // les embeddings par getEmbeddingScale() : la règle de dérivation
                // en chaîne impose le même facteur ici (dL/dTable = dL/dScalé × s).
                INDArray gradForToken = gradOutput.get(NDArrayIndex.point(b), NDArrayIndex.point(s), NDArrayIndex.all());
                gradForToken = gradForToken.mul(getEmbeddingScale());

                // Accumuler le gradient dans embeddingGradients
                embeddingGradients.getRow(tokenId).addi(gradForToken);
            }
        }
    }

    /**
     * Obtient les gradients accumulés des embeddings.
     *
     * @return Liste contenant la matrice de gradients des embeddings.
     */
    public List<INDArray> getGradients() {
        return Collections.singletonList(embeddingGradients);
    }


    /**
     * Réinitialise les gradients accumulés.
     */
    public void resetGradients() {
        embeddingGradients.assign(0.0);
    }
    
    // Initialisation des tokens spéciaux avec des IDs fixes
    private void initializeSpecialTokens() {
        addSpecialToken(PAD_TOKEN);    
        addSpecialToken(UNK_TOKEN);    
        addSpecialToken(START_TOKEN);  
        addSpecialToken(END_TOKEN);    
    }

    // Imprimer le vocabulaire avec le mapping token-ID
    public void printVocabulary() {
        System.out.println("Vocabulaire du Tokenizer:");
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            System.out.printf("Token: '%s' -> ID: %d%n", entry.getKey(), entry.getValue());
        }
    }

    // Initialisation des embeddings avec WordVectors
    public void initializeEmbeddings(WordVectors wordVectors) { // Rendue publique
        if (wordVectors == null) {
            throw new IllegalArgumentException("WordVectors ne peut pas être null.");
        }
        
        this.pretrainedEmbeddings = Nd4j.zeros(vocabSize, embeddingSize);
        
        // Calculer le vecteur moyen pour les tokens inconnus
        INDArray meanVector = calculateMeanVector(wordVectors);
        
        // Initialiser les embeddings des tokens spéciaux
        pretrainedEmbeddings.putRow(getPadTokenId(), Nd4j.zeros(embeddingSize)); // <PAD> est généralement un vecteur nul
        pretrainedEmbeddings.putRow(getUnkTokenId(), meanVector);
        // <START>/<END> = moyenne + bruit ASSORTI au contenu. L'ancien bruit
        // randn*0.1 avait une norme (~1.7) 50x supérieure aux embeddings word2vec
        // (~0.03) : une fois dés-zeroïsés, ils saturaient l'attention en one-hot
        // et tuaient tout gradient. std ~2e-3 = std des coords word2vec.
        double specialNoiseStd = 0.002;
        pretrainedEmbeddings.putRow(getStartTokenId(), meanVector.add(Nd4j.randn(1, embeddingSize).mul(specialNoiseStd)));
        pretrainedEmbeddings.putRow(getEndTokenId(), meanVector.add(Nd4j.randn(1, embeddingSize).mul(specialNoiseStd)));
        
        // Copier les embeddings pour les autres tokens
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            String token = entry.getKey();
            int id = entry.getValue();
            if (isSpecialToken(token)) continue;
            if (wordVectors.hasWord(token)) {
                pretrainedEmbeddings.putRow(id, wordVectors.getWordVectorMatrix(token));
                System.out.println("Embedding pour le token '" + token + "' initialisé avec WordVectors.");
            } else {
                pretrainedEmbeddings.putRow(id, meanVector);
                System.out.println("Embedding pour le token '" + token + "' initialisé avec le vecteur moyen.");
            }
        }
    }

    // Initialisation des embeddings avec des vecteurs aléatoires (pour les tests ou l'utilisation sans WordVectors)
    public void initializeEmbeddings() { // Rendue publique
        this.pretrainedEmbeddings = Nd4j.randn(vocabSize, embeddingSize).divi(Math.sqrt(embeddingSize));
        this.pretrainedEmbeddings.putRow(getPadTokenId(), Nd4j.zeros(embeddingSize)); // <PAD> est un vecteur nul
        this.pretrainedEmbeddings.putRow(getUnkTokenId(), Nd4j.zeros(1, embeddingSize));
        this.pretrainedEmbeddings.putRow(getStartTokenId(), Nd4j.zeros(1, embeddingSize));
        this.pretrainedEmbeddings.putRow(getEndTokenId(), Nd4j.zeros(1, embeddingSize));
    }
    /**
     * Initialise les embeddings avec une matrice identité étendue.
     */
    public void initializeEmbeddingEye() {
        // Créer une matrice identité étendue de taille [vocabSize, dModel]
        // Pour dModel=4 et vocabSize=6, cela pourrait ressembler à :
        // [[1, 0, 0, 0],
        //  [0, 1, 0, 0],
        //  [0, 0, 1, 0],
        //  [0, 0, 0, 1],
        //  [1, 0, 0, 0],
        //  [0, 1, 0, 0]]
        INDArray eye = NDArrayUtils.eye(embeddingSize, embeddingSize);
        INDArray extendedEye = Nd4j.zeros(vocabSize, embeddingSize);
        for(int i = 0; i < vocabSize; i++) {
            extendedEye.putRow(i, eye.getRow(i % embeddingSize));
        }
        this.pretrainedEmbeddings = extendedEye;
    }

    // Calcul du vecteur moyen pour les WordVectors
    private INDArray calculateMeanVector(WordVectors wordVectors) {
        INDArray sum = Nd4j.zeros(embeddingSize);
        int count = 0;
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            sum.addi(wordVectors.getWordVectorMatrix(word));
            count++;
        }
        return sum.div(count);
    }

    // Vérification des tokens spéciaux
    private boolean isSpecialToken(String token) {
        return token.equals(PAD_TOKEN) || token.equals(UNK_TOKEN) || 
               token.equals(START_TOKEN) || token.equals(END_TOKEN);
    }

    // Ajout de tokens spéciaux avec des IDs fixes
    private void addSpecialToken(String token) {
        int id = tokenToId.size();
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    // Ajout de tokens non spéciaux et initialisation de leur embedding
    public void addToken(String token) { // Rendue publique
        if (!tokenToId.containsKey(token)) {
            int id = tokenToId.size();
            tokenToId.put(token, id);
            idToToken.put(id, token);
            // Embeddings ne sont pas modifiés ici; ils doivent être initialisés ultérieurement
            vocabSize = tokenToId.size();
        }
    }

    // Conversion de texte en tokens avec ajout des tokens spéciaux et padding
    public List<String> tokenize(String text) {
        List<String> tokenList = new ArrayList<>();
        tokenList.add(START_TOKEN);
        
        // Tokeniser le texte en ignorant les tokens spéciaux
        String[] words = text.split("\\s+");
        for (String word : words) {
            if (!isSpecialToken(word)) {
                tokenList.add(word);
            }
        }
        
        tokenList.add(END_TOKEN);
        
        // Padding
        while (tokenList.size() < maxSequenceLength) {
            tokenList.add(PAD_TOKEN);
        }
        
        return tokenList;
    }

    // Conversion de tokens en IDs
    public List<Integer> tokensToIds(List<String> tokens) {
        return tokens.stream()
                     .map(token -> {
                         int id = tokenToId.getOrDefault(token, getUnkTokenId());
                         updateFrequency(id);
                         return id;
                     })
                     .collect(Collectors.toList());
    }

    // Conversion d'une liste d'IDs en chaîne de tokens
    public String idsToTokens(List<Integer> ids) {
        return ids.stream()
                  .map(id -> idToToken.getOrDefault(id, UNK_TOKEN))
                  .collect(Collectors.joining(" "));
    }

    /**
     * Convertit une liste d'IDs de tokens en une liste de tokens.
     *
     * @param tokenIds Liste d'IDs de tokens.
     * @return Liste de tokens correspondants.
     */
    public List<String> idsToListTokens(List<Integer> tokenIds) {
        return tokenIds.stream()
                       .map(id -> idToToken.getOrDefault(id, UNK_TOKEN))
                       .collect(Collectors.toList());
    }

    // Méthodes pour obtenir les IDs des tokens spéciaux
    public int getPadTokenId() {
        return tokenToId.get(PAD_TOKEN);
    }

    public Map<Integer, String> getIdToTokenMap() {
        Map<Integer, String> idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            idToToken.put(entry.getValue(), entry.getKey());
        }
        return idToToken;
    }

    public Map<String, Integer> getTokenToId(){
        return tokenToId;
    }

    public int getUnkTokenId() {
        return tokenToId.get(UNK_TOKEN);
    }

    public int getStartTokenId() {
        return tokenToId.get(START_TOKEN);
    }

    public int getEndTokenId() {
        return tokenToId.get(END_TOKEN);
    }

    // Taille du vocabulaire
    public int getVocabSize() {
        return vocabSize;
    }

    // Gestion des embeddings
    public INDArray getPretrainedEmbeddings() {
        return pretrainedEmbeddings;
    }

    public INDArray getWeights() {
        return pretrainedEmbeddings;
    }
    
    public void setPretrainedEmbeddings(INDArray embeddings) {
        this.pretrainedEmbeddings = embeddings;
    }

    public boolean hasCalculatedEmbedding(int tokenId) {
        INDArray embedding = getPretrainedEmbeddings().getRow(tokenId);
        // Définir votre propre logique pour déterminer si l'embedding est calculé
        // Par exemple, vérifier si la moyenne de l'embedding est non zéro
        double mean = embedding.meanNumber().doubleValue();
        return mean != 0.0;
    }

    /**
     * Obtient les embeddings du Tokenizer.
     *
     * @return Une liste contenant la matrice d'embeddings.
     */
    public List<INDArray> getEmbeddings() {
        return Collections.singletonList(pretrainedEmbeddings);
    }


    /**
     * Initialise les gradients pour les embeddings.
     */
    public void initializeEmbeddingGradients() {
        embeddingGradients = Nd4j.zeros(vocabSize, embeddingSize);
    }


    /**
     * Applique les gradients accumulés aux embeddings.
     * Ceci devrait être appelé après backward() et avant la mise à jour des paramètres.
     *
     * @param learningRate Le taux d'apprentissage.
     */
    public void applyGradients(float learningRate) {
        // Mettre à jour les embeddings avec le gradient (descente de gradient simple)
        pretrainedEmbeddings.subi(embeddingGradients.mul(learningRate));
        
        // Réinitialiser les gradients après mise à jour
        embeddingGradients.assign(0.0);
    }

    public int getFrequency(int tokenId) {
        return tokenFrequencies.getOrDefault(tokenId, 0);
    }
    
    // Dans la méthode encode ou là où vous traitez les tokens
    public void updateFrequency(int tokenId) {
        tokenFrequencies.merge(tokenId, 1, Integer::sum);
    }

    public String idToToken(int id) {
        return idToToken.getOrDefault(id, "<UNK>");
    }

    public int getTokenId(String string) {
        return tokenToId.getOrDefault(string, getUnkTokenId());
    }

}
