package RN.transformer;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Tokenizer implements Serializable {
    private static final long serialVersionUID = -1691008595018974489L;
    private Map<String, Integer> tokenToId;
    private Map<Integer, String> idToToken;
    private int vocabSize;
    private INDArray pretrainedEmbeddings;

    // Tokens spéciaux
    private static final String PAD_TOKEN = "<PAD>";
    private static final String UNK_TOKEN = "<UNK>";
    private static final String START_TOKEN = "<START>";
    private static final String END_TOKEN = "<END>";
    
    // Les embeddings pour les tokens spéciaux seront initialisés de manière spécifique
    private static final int EMBEDDING_SIZE = 300; // dModel

    public Tokenizer(WordVectors wordVectors) {
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        
        // Étape 1: Initialiser les tokens spéciaux en premier
        // Cela garantit que leurs IDs sont constants (0, 1, 2, 3)
        initializeSpecialTokens();
        
        // Étape 2: Ajouter tous les mots du vocabulaire Word2Vec
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            if (!tokenToId.containsKey(word)) {
                addToken(word);
            }
        }
        
        this.vocabSize = tokenToId.size();
        
        // Étape 3: Créer la matrice d'embeddings avec les tokens spéciaux
        initializeEmbeddings(wordVectors);
    }
    
    public Tokenizer(List<String> words) {
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        
        // Étape 1: Initialiser les tokens spéciaux en premier
        // Cela garantit que leurs IDs sont constants (0, 1, 2, 3)
        initializeSpecialTokens();
        
        // Étape 2: Ajouter tous les mots du vocabulaire Word2Vec
        for (String word : words) {
            if (!tokenToId.containsKey(word)) {
                addToken(word);
            }
        }
        
        this.vocabSize = tokenToId.size();
        
        // Étape 3: Créer la matrice d'embeddings avec les tokens spéciaux
//        initializeEmbeddings(words);
    }

    private void initializeSpecialTokens() {
        // Les tokens spéciaux sont toujours ajoutés dans le même ordre
        addSpecialToken(PAD_TOKEN);    // ID = 0
        addSpecialToken(UNK_TOKEN);    // ID = 1
        addSpecialToken(START_TOKEN);  // ID = 2
        addSpecialToken(END_TOKEN);    // ID = 3
    }

    private void initializeEmbeddings(WordVectors wordVectors) {
        // Créer une nouvelle matrice d'embeddings avec la taille du vocabulaire complet
        pretrainedEmbeddings = Nd4j.zeros(vocabSize, EMBEDDING_SIZE);
        
        // Calculer le vecteur moyen pour l'initialisation des tokens spéciaux
        INDArray meanVector = calculateMeanVector(wordVectors);
        
        // Initialiser les embeddings des tokens spéciaux
        // PAD_TOKEN: vecteur de zéros (déjà initialisé par défaut)
        // UNK_TOKEN: vecteur moyen
        pretrainedEmbeddings.putRow(getUnkTokenId(), meanVector);
        // START_TOKEN: vecteur moyen + bruit gaussien
        pretrainedEmbeddings.putRow(getStartTokenId(), meanVector.add(Nd4j.randn(1, EMBEDDING_SIZE).mul(0.1)));
        // END_TOKEN: vecteur moyen + bruit gaussien
        pretrainedEmbeddings.putRow(getEndTokenId(), meanVector.add(Nd4j.randn(1, EMBEDDING_SIZE).mul(0.1)));
        
        // Copier les embeddings pour tous les autres tokens
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            String token = entry.getKey();
            int id = entry.getValue();
            
            // Ignorer les tokens spéciaux déjà traités
            if (isSpecialToken(token)) continue;
            
            if (wordVectors.hasWord(token)) {
                INDArray wordVector = wordVectors.getWordVectorMatrix(token);
                pretrainedEmbeddings.putRow(id, wordVector);
            } else {
                // Utiliser le vecteur moyen pour les mots inconnus
                pretrainedEmbeddings.putRow(id, meanVector);
            }
        }
    }

    private INDArray calculateMeanVector(WordVectors wordVectors) {
        INDArray sum = Nd4j.zeros(EMBEDDING_SIZE);
        int count = 0;
        
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            sum.addi(wordVectors.getWordVectorMatrix(word));
            count++;
        }
        
        return sum.div(count);
    }

    private boolean isSpecialToken(String token) {
        return token.equals(PAD_TOKEN) || token.equals(UNK_TOKEN) || 
               token.equals(START_TOKEN) || token.equals(END_TOKEN);
    }

    private void addSpecialToken(String token) {
        int id = tokenToId.size();
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    private void addToken(String token) {
        if (!tokenToId.containsKey(token)) {
            int id = tokenToId.size();
            tokenToId.put(token, id);
            idToToken.put(id, token);
        }
    }

    public List<String> tokenize(String text) {
        // Cette regex simple sépare les mots et la ponctuation, ce qui est une amélioration par rapport à la séparation par espace.
        // Pour des règles plus complexes, envisagez d'utiliser une librairie de tokenisation spécialisée.
        String[] tokens = text.split("\\s+|(?=\\p{Punct})|(?<=\\p{Punct})");
        List<String> tokenList = new ArrayList<>();
        for (String token : tokens) {
            if (!token.trim().isEmpty()) { // Ignorer les chaînes vides
                tokenList.add(token);
            }
        }
        return tokenList;
    }


    private boolean isPunctuation(String token) {
        // Une vérification simple de la ponctuation basée sur regex; ajustez selon vos besoins
        return token.matches("\\p{Punct}");
    }

    public List<Integer> tokensToIds(List<String> tokens) {
        return tokens.stream()
                .map(token -> tokenToId.getOrDefault(token, tokenToId.get(UNK_TOKEN)))
                .collect(Collectors.toList());
    }

    public String idsToTokens(List<Integer> ids) {
        return ids.stream()
                .map(id -> idToToken.getOrDefault(id, UNK_TOKEN))
                .collect(Collectors.joining(" "));
    }

    // Nouvelles méthodes pour gérer les tokens spéciaux
    public int getPadTokenId() {
        return tokenToId.get(PAD_TOKEN);
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

    public int getVocabSize() {
        return vocabSize;
    }

    public String getToken(int id) {
        return idToToken.getOrDefault(id, UNK_TOKEN);
    }
    
    public INDArray getPretrainedEmbeddings() {
        return pretrainedEmbeddings;
    }
    
}



