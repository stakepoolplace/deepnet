package RN.transformer;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Tokenizer {
    private Map<String, Integer> tokenToIdMap;
    private Map<Integer, String> idToTokenMap;
    private int unkTokenId; // Identifiant pour les tokens inconnus

    public Tokenizer() {
        this.tokenToIdMap = new HashMap<>();
        this.idToTokenMap = new HashMap<>();
        this.unkTokenId = -1; // Initialiser avec une valeur pour les tokens inconnus
        initializeVocabulary();
    }

    private void initializeVocabulary() {
        // Initialiser votre vocabulaire ici
        // Exemple: ajouter un token inconnu
        addTokenToVocabulary("[UNK]");
        unkTokenId = tokenToIdMap.get("[UNK]");

        // Ajouter d'autres tokens au vocabulaire
        addTokenToVocabulary("ceci");
        addTokenToVocabulary("est");
        addTokenToVocabulary("un");
        addTokenToVocabulary("exemple");
        // Continuer pour le reste du vocabulaire
    }

    private void addTokenToVocabulary(String token) {
        if (!tokenToIdMap.containsKey(token)) {
            int newId = tokenToIdMap.size();
            tokenToIdMap.put(token, newId);
            idToTokenMap.put(newId, token);
        }
    }

    public List<String> tokenize(List<String> words) {
    	
        // Simple tokenization basée sur l'espace
        //String[] tokens = text.split("\\s+");
    	
        List<String> tokenList = new ArrayList<>();
        for (String token : words) {
            tokenList.add(token.toLowerCase()); // Convertir en minuscule pour la simplicité
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
            text.append(idToTokenMap.getOrDefault(id, "[UNK]")).append(" ");
        }
        return text.toString().trim();
    }
}
