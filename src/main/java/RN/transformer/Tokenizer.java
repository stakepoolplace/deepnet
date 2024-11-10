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
    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    private int vocabSize;
    private INDArray pretrainedEmbeddings;
    private final int embeddingSize;

    // Tokens spéciaux
    private static final String PAD_TOKEN = "<PAD>";
    private static final String UNK_TOKEN = "<UNK>";
    private static final String START_TOKEN = "<START>";
    private static final String END_TOKEN = "<END>";
    
    // Constructeur utilisant des WordVectors
    public Tokenizer(WordVectors wordVectors, int embeddingSize) {
        this.embeddingSize = embeddingSize;
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        
        // Initialiser les tokens spéciaux
        initializeSpecialTokens();
        
        // Ajouter tous les mots du vocabulaire Word2Vec
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            addToken(word);
        }
        
        this.vocabSize = tokenToId.size();
        
        // Initialiser les embeddings avec WordVectors
        initializeEmbeddings(wordVectors);
    }
    
    // Constructeur utilisant une liste de mots
    public Tokenizer(List<String> words, int embeddingSize) {
        this.embeddingSize = embeddingSize;
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        
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


    public INDArray tokensToINDArray(List<String> tokens) {
        // System.out.println("Tokens to convert: " + tokens);
        int[] ids = tokens.stream()
                          .mapToInt(token -> tokenToId.getOrDefault(token, getUnkTokenId()))
                          .toArray();
        // System.out.println("Token to ID Mapping in tokensToINDArray: " + Arrays.toString(ids));
        
        // Utiliser createFromArray et reshape pour garantir la forme [1, N]
        INDArray arr = Nd4j.createFromArray(ids).castTo(DataType.INT32).reshape(1, ids.length);
        // System.out.println("Shape of INDArray: " + Arrays.toString(arr.shape()));
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
        
        // Calculer dModel en obtenant la taille de la dernière dimension de batchEmbeddings
        int dModel = (int) batchEmbeddings.shape()[1];

        // Reshaper en [batchSize, seqLength, dModel]
        batchEmbeddings = batchEmbeddings.reshape(batchSize, seqLength, dModel);
        
        return batchEmbeddings;
    }
    
    // Initialisation des tokens spéciaux avec des IDs fixes
    private void initializeSpecialTokens() {
        addSpecialToken(PAD_TOKEN);    
        addSpecialToken(UNK_TOKEN);    
        addSpecialToken(START_TOKEN);  
        addSpecialToken(END_TOKEN);    
    }

    // Initialisation des embeddings avec WordVectors
    private void initializeEmbeddings(WordVectors wordVectors) {
        pretrainedEmbeddings = Nd4j.zeros(vocabSize, embeddingSize);
        
        // Calculer le vecteur moyen pour les tokens spéciaux
        INDArray meanVector = calculateMeanVector(wordVectors);
        
        // Initialiser les embeddings des tokens spéciaux
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
            } else {
                pretrainedEmbeddings.putRow(id, meanVector);
            }
        }
    }

    // Initialisation des embeddings pour les tests
    private void initializeEmbeddings() {
        pretrainedEmbeddings = Nd4j.randn(vocabSize, embeddingSize).divi(Math.sqrt(embeddingSize));
        pretrainedEmbeddings.putRow(getUnkTokenId(), Nd4j.randn(1, embeddingSize));
        pretrainedEmbeddings.putRow(getStartTokenId(), Nd4j.randn(1, embeddingSize));
        pretrainedEmbeddings.putRow(getEndTokenId(), Nd4j.randn(1, embeddingSize));
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

    // Ajout de tokens spéciaux
    private void addSpecialToken(String token) {
        int id = tokenToId.size();
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    // Ajout de tokens non spéciaux
    private void addToken(String token) {
        if (!tokenToId.containsKey(token)) {
            int id = tokenToId.size();
            tokenToId.put(token, id);
            idToToken.put(id, token);
        }
    }

    // Conversion de texte en tokens
    public List<String> tokenize(String text) {
        String[] tokens = text.split("\\s+|(?=\\p{Punct})|(?<=\\p{Punct})");
        return Arrays.stream(tokens)
                     .filter(token -> !token.trim().isEmpty())
                     .collect(Collectors.toList());
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
}
