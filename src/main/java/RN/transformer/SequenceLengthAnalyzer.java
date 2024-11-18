package RN.transformer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class SequenceLengthAnalyzer {
    public static void main(String[] args) {
        // Supposons que vous avez une méthode pour obtenir toutes les phrases de votre jeu de données
        List<String> allSentences = getAllSentences();

        // Tokenizer sans padding/troncature pour l'analyse
        Tokenizer tokenizer = new Tokenizer(/* paramètres appropriés */);

        // Collecter les longueurs
        List<Integer> lengths = allSentences.stream()
            .map(sentence -> tokenizer.tokenize(sentence).size())
            .collect(Collectors.toList());

        // Calculer les statistiques
        double mean = lengths.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        int max = lengths.stream().mapToInt(Integer::intValue).max().orElse(0);
        double percentile95 = calculatePercentile(lengths, 95);

        System.out.println("Moyenne de la longueur des séquences : " + mean);
        System.out.println("Longueur maximale des séquences : " + max);
        System.out.println("95ème percentile de la longueur des séquences : " + percentile95);
    }

    // Implémentez ces méthodes selon vos besoins
    private static List<String> getAllSentences() {
        // Retourner toutes les phrases de votre jeu de données
        return Arrays.asList("exemple phrase 1", "exemple phrase 2", ...);
    }

    private static double calculatePercentile(List<Integer> lengths, double percentile) {
        int index = (int) Math.ceil(percentile / 100.0 * lengths.size());
        List<Integer> sorted = lengths.stream().sorted().collect(Collectors.toList());
        return sorted.get(index - 1);
    }
}
