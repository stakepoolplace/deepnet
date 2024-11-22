// Tokenizer.java
package RN.transformer;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Tokenizer implements Serializable {
    private static final long serialVersionUID = -1691008595018974489L;
    
    // Utiliser LinkedHashMap pour préserver l'ordre d'insertion
    private final LinkedHashMap<String, Integer> tokenToId;
    private final LinkedHashMap<Integer, String> idToToken;
    private int vocabSize;
    private INDArray pretrainedEmbeddings;
    private final int embeddingSize;
    private final int maxSequenceLength;


    // Tokens spéciaux
    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final String START_TOKEN = "<START>";
    public static final String END_TOKEN = "<END>";
    
    // WordVectors membre
    private WordVectors wordVectors;
    
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
    }

    /**
     * Convertit une liste de tokens en INDArray d'IDs.
     */
    public INDArray tokensToINDArray(List<String> tokens) {
        int[] ids = tokens.stream()
                          .mapToInt(token -> tokenToId.getOrDefault(token, getUnkTokenId()))
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
        
        // Récupérer les embeddings: [batchSize * seqLength, dModel]
        INDArray batchEmbeddings = getPretrainedEmbeddings().getRows(tokenIds); 
        
        // Reshaper en [batchSize, seqLength, dModel]
        batchEmbeddings = batchEmbeddings.reshape(batchSize, seqLength, embeddingSize);

        return batchEmbeddings;
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
        pretrainedEmbeddings.putRow(getStartTokenId(), meanVector.add(Nd4j.randn(1, embeddingSize).mul(0.1)));
        pretrainedEmbeddings.putRow(getEndTokenId(), meanVector.add(Nd4j.randn(1, embeddingSize).mul(0.1)));
        
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
        pretrainedEmbeddings.putRow(getPadTokenId(), Nd4j.zeros(embeddingSize)); // <PAD> est un vecteur nul
        pretrainedEmbeddings.putRow(getUnkTokenId(), Nd4j.zeros(1, embeddingSize));
        pretrainedEmbeddings.putRow(getStartTokenId(), Nd4j.zeros(1, embeddingSize));
        pretrainedEmbeddings.putRow(getEndTokenId(), Nd4j.zeros(1, embeddingSize));
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
        String[] tokens = text.split("\\s+|(?=\\p{Punct})|(?<=\\p{Punct})");
        List<String> tokenList = new ArrayList<>();
        
        // Ajouter le token <START>
        tokenList.add(START_TOKEN);
        
        // Ajouter les tokens de la phrase
        tokenList.addAll(Arrays.stream(tokens)
                             .filter(token -> !token.trim().isEmpty())
                             .collect(Collectors.toList()));
        
        // Ajouter le token <END>
        tokenList.add(END_TOKEN);
        
        
        // Ajouter le padding <PAD> si nécessaire
        while (tokenList.size() < maxSequenceLength) {
            tokenList.add(PAD_TOKEN);
        }
        
        // Troncature si la séquence dépasse la longueur maximale
        if (tokenList.size() > maxSequenceLength) {
            tokenList = tokenList.subList(0, maxSequenceLength);
        }
        
        return tokenList;
    }

    // Conversion de tokens en IDs
    public List<Integer> tokensToIds(List<String> tokens) {
        return tokens.stream()
                     .map(token -> tokenToId.getOrDefault(token, getUnkTokenId()))
                     .collect(Collectors.toList());
    }

    // Conversion d'une liste d'IDs en chaîne de tokens
    public String idsToTokens(List<Integer> ids) {
        return ids.stream()
                  .map(id -> idToToken.getOrDefault(id, UNK_TOKEN))
                  .collect(Collectors.joining(" "));
    }

    // Méthodes pour obtenir les IDs des tokens spéciaux
    public int getPadTokenId() {
        return tokenToId.get(PAD_TOKEN);
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
}
