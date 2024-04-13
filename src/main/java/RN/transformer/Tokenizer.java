package RN.transformer;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

public class Tokenizer {
    private Map<String, Integer> tokenToIdMap = new HashMap<>();
    private Map<Integer, String> idToTokenMap = new HashMap<>();
    private WordVectors wordVectors;
    private int unkTokenId; // Identifiant pour les tokens inconnus
    private int padTokenId; // Identifiant pour les tokens de padding
    
    public Tokenizer() {
	}
    
    public Tokenizer(WordVectors wordVectors) throws IOException {
    	
        // Synchroniser le vocabulaire avec Word2Vec
        synchronizeVocabularyWithWord2Vec(wordVectors);
    }


    private void synchronizeVocabularyWithWord2Vec(WordVectors wordVectors) {
        int index = 0;
        Collection<String> words = wordVectors.vocab().words();
        for (String word : words) {
            tokenToIdMap.put(word, index);
            idToTokenMap.put(index, word);
            index++;
        }

        // Gérer le token inconnu
        String unkToken = "[UNK]";
        unkTokenId = index; // Attribuer l'identifiant suivant au token inconnu
        tokenToIdMap.put(unkToken, unkTokenId);
        idToTokenMap.put(unkTokenId, unkToken);

        // Ajouter un token de padding
        String padToken = "[PAD]";
        padTokenId = ++index; // Attribuer le prochain identifiant au token de padding
        tokenToIdMap.put(padToken, padTokenId);
        idToTokenMap.put(padTokenId, padToken);
    }


    public int getPadTokenId() {
        return padTokenId; // Retourner l'identifiant du token de padding
    }	
	
    public String getToken(int tokenId) {
        return idToTokenMap.getOrDefault(tokenId, "[UNK]");
    }
    
    public List<String> tokenize(String text) {
        // Cette regex simple sépare les mots et la ponctuation, ce qui est une amélioration par rapport à la séparation par espace.
        // Pour des règles plus complexes, envisagez d'utiliser une librairie de tokenisation spécialisée.
        String[] tokens = text.split("\\s+|(?=\\p{Punct})|(?<=\\p{Punct})");
        List<String> tokenList = new ArrayList<>();
        for (String token : tokens) {
            if (!token.trim().isEmpty()) { // Ignorer les chaînes vides
                tokenList.add(token.toLowerCase()); // Convertir en minuscule pour la simplicité
            }
        }
        return tokenList;
    }


    public List<Integer> tokensToIds(List<String> tokens) {
        List<Integer> ids = new ArrayList<>();
        for (String token : tokens) {
            ids.add(tokenToIdMap.getOrDefault(token, unkTokenId));
        }
        return ids;
    }

    
    public String idsToTokens(List<Integer> ids) {
        StringBuilder text = new StringBuilder();
        for (int id : ids) {
            String token = idToTokenMap.getOrDefault(id, "[UNK]");
            if (text.length() > 0) text.append(" ");
            text.append(token);
        }
        return text.toString().trim();
    }


    private boolean isPunctuation(String token) {
        // Une vérification simple de la ponctuation basée sur regex; ajustez selon vos besoins
        return token.matches("\\p{Punct}");
    }

}
