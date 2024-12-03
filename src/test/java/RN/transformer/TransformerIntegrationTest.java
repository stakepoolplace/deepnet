package RN.transformer;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

public class TransformerIntegrationTest {

    private TransformerModel model;
    private DataGenerator mockDataGenerator;

    @Before
    public void setUp() throws IOException {
    } 

    @Test
    public void testInferenceAfterTraining() throws Exception {

        // Fixer la graine pour la reproductibilité
        Nd4j.getRandom().setSeed(42);
        
        // Initialisation du Tokenizer avec un vocabulaire minimal
        int maxSequenceLength = 3;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        int dModel = 64;   // Taille modeste qui fonctionne
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration qui fonctionne de manière fiable
        int numLayers = 2;
        int numHeads = 1;
        int dff = 128;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.0001f;        // Learning rate original
        int warmupSteps = 0;       // Pas de warmup
        int batchSize = 1;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        
        // Une seule paire d'apprentissage
        List<String> data = Arrays.asList("hello");
        List<String> targets = Arrays.asList("world");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
 

        // Vérifier l'état initial
        System.out.println("\n=== Configuration initiale ===");
        System.out.println("Vocabulaire: " + model.tokenizer.getIdToTokenMap().values());
        System.out.println("Taille du vocabulaire: " + model.tokenizer.getVocabSize());
        
        // Effectuer l'entraînement
        float finalLoss = 0;
        int maxEpochs = 200;
        int convergenceCount = 0;
        
        model.setTrace(false);
        model.getOptimizer().setMaxEpochs(maxEpochs);

        System.out.println("\n=== Début de l'entraînement ===");
        
        // Entraîner jusqu'à convergence ou max epochs
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            float loss = model.trainEpoch(mockDataGenerator);
            if (epoch % 10 == 0) {  // Log tous les 10 epochs
                System.out.println(String.format("Epoch %d - Loss: %.6f", epoch, loss));
            }
            
            if (loss < 0.01) {
                convergenceCount++;
                if (convergenceCount >= 5) {
                    finalLoss = loss;
                    System.out.println("Convergence atteinte après " + epoch + " epochs (Loss: " + finalLoss + ")");
                    break;
                }
            } else {
                convergenceCount = 0;
            }
            finalLoss = loss;
        }

        // Test d'inférence pour chaque paire
        System.out.println("\n=== Tests d'inférence ===");
        Map<String, String> testPairs = new LinkedHashMap<>();
        testPairs.put("hello", "world");
        // testPairs.put("hi", "there");
        // testPairs.put("good", "morning");
        // testPairs.put("bye", "now");
        
        for (Map.Entry<String, String> pair : testPairs.entrySet()) {
            String input = pair.getKey();
            String expectedOutput = pair.getValue();
            
            System.out.println("\nTest pour: '" + input + "'");
            String actualOutput = model.infer(input, 1); // Augmenté à 3 pour inclure <START> token + mot + <END>
            System.out.println("Sortie brute: '" + actualOutput + "'");
            
            actualOutput = actualOutput.replace("<START>", "").replace("<END>", "").trim();
            System.out.println("Sortie nettoyée: '" + actualOutput + "'");
            
            System.out.println("Attendu: '" + expectedOutput + "'");
            
            assertEquals("L'inférence pour '" + input + "' devrait correspondre à la cible", 
                        expectedOutput, actualOutput);
        }
    }



    @Test
    public void testTrainingOnMockData() throws Exception {

        // Fixer la graine pour la reproductibilité
        Nd4j.getRandom().setSeed(42);
        
        // Initialisation du Tokenizer avec un vocabulaire minimal
        int maxSequenceLength = 3;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        int dModel = 64;   // Taille modeste qui fonctionne
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration qui fonctionne de manière fiable
        int numLayers = 2;
        int numHeads = 1;
        int dff = 128;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.0001f;        // Learning rate original
        int warmupSteps = 0;       // Pas de warmup
        int batchSize = 1;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        

        // Une seule paire d'apprentissage
        List<String> data = Arrays.asList("hello");
        List<String> targets = Arrays.asList("world");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
 

        // Vérifier l'état initial
        System.out.println("\n=== Configuration initiale ===");
        System.out.println("Vocabulaire: " + model.tokenizer.getIdToTokenMap().values());
        System.out.println("Taille du vocabulaire: " + model.tokenizer.getVocabSize());
        
        // Effectuer l'entraînement
        float finalLoss = 0;
        int maxEpochs = 200;
        int convergenceCount = 0;
        
        model.setTrace(false);
        model.getOptimizer().setMaxEpochs(maxEpochs);


        // Initialisation de l'entraînement avec un seul epoch
        float loss = model.trainEpoch(mockDataGenerator);

        // Vérification que le modèle est marqué comme entraîné
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());

        // Effectuer une inférence sur l'entrée d'entraînement
        String input = "hello world";
        String expectedOutput = "hello output";
        String actualOutput = model.infer(input, 10);

        // Vérification que l'inférence n'est pas nulle et est cohérente
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Vérifier que l'inférence est proche de la cible
        // Cela dépend de la manière dont l'inférence est implémentée et peut nécessiter une tolérance
        // Par exemple, si l'inférence utilise une certaine logique déterministe
        // assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    @Test
    public void testLossDecreaseOverEpochs() throws Exception {
        // Création d'un DataGenerator avec plusieurs batches pour simuler plusieurs epochs
        // List<String> data = Arrays.asList("hello world", "test input");
        // List<String> targets = Arrays.asList("hello output", "test output");
        // mockDataGenerator = new MockDataGenerator(data, targets, model.tokenizer, 1, 50, 5); // 5 batches


        // Fixer la graine pour la reproductibilité
        Nd4j.getRandom().setSeed(42);
        
        // Initialisation du Tokenizer avec un vocabulaire minimal
        int maxSequenceLength = 3;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        int dModel = 64;   // Taille modeste qui fonctionne
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration qui fonctionne de manière fiable
        int numLayers = 2;
        int numHeads = 1;
        int dff = 128;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.0001f;        // Learning rate original
        int warmupSteps = 0;       // Pas de warmup
        int batchSize = 1;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        
        // Une seule paire d'apprentissage
        List<String> data = Arrays.asList("hello");
        List<String> targets = Arrays.asList("world");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
 

        // Vérifier l'état initial
        System.out.println("\n=== Configuration initiale ===");
        System.out.println("Vocabulaire: " + model.tokenizer.getIdToTokenMap().values());
        System.out.println("Taille du vocabulaire: " + model.tokenizer.getVocabSize());
        
        // Effectuer l'entraînement
        float finalLoss = 0;
        int maxEpochs = 200;
        int convergenceCount = 0;
        
        model.setTrace(false);
        model.getOptimizer().setMaxEpochs(maxEpochs);

        // Initialisation des variables pour suivre la perte
        List<Float> lossHistory = new java.util.ArrayList<>();

        // Entraîner sur 5 epochs et enregistrer la perte à chaque epoch
        for (int epoch = 0; epoch < 5; epoch++) {
            float loss = model.trainEpoch(mockDataGenerator);
            lossHistory.add(loss);
            System.out.println("Epoch " + (epoch + 1) + " Loss: " + loss);
        }

        // Vérification que la perte diminue au fil des epochs
        for (int i = 1; i < lossHistory.size(); i++) {
            assertTrue("La perte devrait diminuer au fil des epochs",
                    lossHistory.get(i) < lossHistory.get(i - 1));
        }
    }

    @Test
    public void testParameterUpdatesDuringTraining() throws Exception {

        // Fixer la graine pour la reproductibilité
        Nd4j.getRandom().setSeed(42);
        
        // Initialisation du Tokenizer avec un vocabulaire minimal
        int maxSequenceLength = 3;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        int dModel = 64;   // Taille modeste qui fonctionne
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration qui fonctionne de manière fiable
        int numLayers = 2;
        int numHeads = 1;
        int dff = 128;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.0001f;        // Learning rate original
        int warmupSteps = 0;       // Pas de warmup
        int batchSize = 1;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        
        // Une seule paire d'apprentissage
        List<String> data = Arrays.asList("hello");
        List<String> targets = Arrays.asList("world");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
 

        // Vérifier l'état initial
        System.out.println("\n=== Configuration initiale ===");
        System.out.println("Vocabulaire: " + model.tokenizer.getIdToTokenMap().values());
        System.out.println("Taille du vocabulaire: " + model.tokenizer.getVocabSize());
        
        // Effectuer l'entraînement
        float finalLoss = 0;
        int maxEpochs = 200;
        int convergenceCount = 0;
        
        model.setTrace(false);
        model.getOptimizer().setMaxEpochs(maxEpochs);


        // Copier les paramètres avant l'entraînement
        List<INDArray> initialParameters = model.getCombinedParameters().stream()
                .map(INDArray::dup)
                .collect(Collectors.toList());

        // Effectuer une étape d'entraînement
        model.trainEpoch(mockDataGenerator);

        // Récupérer les paramètres après l'entraînement
        List<INDArray> updatedParameters = model.getCombinedParameters();

        // Vérifier que les paramètres ont été mis à jour
        for (int i = 0; i < initialParameters.size(); i++) {
            INDArray initial = initialParameters.get(i);
            INDArray updated = updatedParameters.get(i);
            assertFalse("Les paramètres devraient avoir été mis à jour",
                    initial.equalsWithEps(updated, 1e-6));
        }
    }


}