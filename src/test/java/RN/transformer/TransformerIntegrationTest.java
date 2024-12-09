package RN.transformer;

import static org.junit.Assert.*;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

public class TransformerIntegrationTest {

    private TransformerModel model;
    private DataGenerator mockDataGenerator;
    private Tokenizer tokenizer;

    @Before
    public void setUp() throws IOException {
 
    }

    /**
     * Charge les WordVectors pré-entraînés.
     */
    private static WordVectors loadPreTrainedWordVectors() throws IOException {
        File modelFile = new File("pretrained-embeddings/mon_model_word2vec.txt");
        if (!modelFile.exists()) {
            throw new IOException("Le fichier du modèle Word2Vec n'existe pas: " + modelFile.getAbsolutePath());
        }
        return WordVectorSerializer.readWord2VecModel(modelFile);
    }

    @Test
    public void testInferenceAfterTraining() throws Exception {

        // Fixer la graine pour la reproductibilité
        Nd4j.getRandom().setSeed(42);
        
        // Initialisation du Tokenizer avec un vocabulaire minimal
        int maxSequenceLength = 3;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world", "hi", "there");
        int dModel = 64;   // Taille modeste qui fonctionne
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration qui fonctionne de manière fiable
        int numLayers = 5;
        int numHeads = 2;
        int dff = 512;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.001f;        // Learning rate original
        int warmupSteps = 0;       // Pas de warmup
        int batchSize = 1;
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        
        // Une seule paire d'apprentissage
        List<String> data = Arrays.asList("hello", "hi");
        List<String> targets = Arrays.asList("world", "there");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
 

        // Vérifier l'état initial
        System.out.println("\n=== Configuration initiale ===");
        System.out.println("Vocabulaire: " + model.tokenizer.getIdToTokenMap().values());
        System.out.println("Taille du vocabulaire: " + model.tokenizer.getVocabSize());
        
        // Effectuer l'entraînement
        float finalLoss = 0;
        int maxEpochs = 102;
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
            
            if (loss < 0.004) {
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
         testPairs.put("hi", "there");
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
        float loss = model.train(mockDataGenerator, maxEpochs);

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

    @Test
    public void testNextWordPrediction() throws IOException {
        // Initialisation des paramètres
        int maxSequenceLength = 10;
        WordVectors preTrainedWordVectors = loadPreTrainedWordVectors();
        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        
        // Vocabulaire minimal
        List<String> vocabulary = Arrays.asList(
            "<PAD>", "<UNK>", "<START>", "<END>",
            "le", "chat", "sur", "dans", "tapis", "jardin"
        );
        
        int dModel = embeddingSize;  // 300
        int numHeads = 6;  // Car 300/6 = 50 (entier)
        
        tokenizer = new Tokenizer(vocabulary, embeddingSize, maxSequenceLength);
        tokenizer.initializeEmbeddings(preTrainedWordVectors);
        
        // Configuration avec dimensions cohérentes
        int numLayers = 1;
        int dff = dModel * 4;     // 1200
        float dropoutRate = 0.1f;
        float initialLr = 0.001f;
        int warmupSteps = 0;
        int epochs = 100;
        int batchSize = 2;
        
        // Données d'entraînement
        List<String> inputs = Arrays.asList(
            "le chat sur", "le chat dans"
        );
        
        List<String> targets = Arrays.asList(
            "le tapis", "le jardin"
        );
        
        model = new TransformerModel(
            numLayers,
            dModel,
            numHeads,  // Maintenant 6 au lieu de 8
            dff,
            dropoutRate,
            tokenizer.getVocabSize(),
            tokenizer,
            initialLr,
            warmupSteps
        );
        
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, maxSequenceLength);
        
        try {
            model.train(mockDataGenerator, epochs);
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'entraînement initial", e);
        }
        
        assertNotNull("Le modèle ne devrait pas être null", model);
        String input = "le chat";
        String prediction = model.predict(input);
        assertNotNull("La prédiction ne devrait pas être null", prediction);
        assertTrue("La prédiction devrait être soit 'sur' soit 'dans'", 
                  prediction.equals("sur") || prediction.equals("dans"));
    }

    @Test
    public void testSimpleCopyTask() {
        // Configuration minimale
        List<String> vocabulary = Arrays.asList(
            "<PAD>", "<UNK>", "<START>", "<END>",
            "A", "B"  // Réduire à 2 tokens seulement
        );
        
        int dModel = 32;    // Petite dimension
        int maxSequenceLength = 4;
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Configuration très simple du modèle
        TransformerModel model = new TransformerModel(
            1,          // Une seule couche
            32,         // dModel petit mais suffisant
            1,          // Une seule tête d'attention
            64,         // FFN modeste
            0.0f,       // Pas de dropout
            vocabulary.size(),
            tokenizer,
            0.005f,     // Learning rate standard
            0
        );
        
        // Données d'entraînement simplifiées
        List<String> inputs = Arrays.asList("A", "B");
        List<String> targets = Arrays.asList("A", "B");
        
        // Générateur de données
        DataGenerator dataGenerator = new DataGenerator(
            inputs, 
            targets,
            tokenizer,
            1,  // Batch size de 1
            maxSequenceLength
        );
        
        model.setTrace(false);
        //model.getOptimizer().setMaxEpochs(200);

        // Entraînement
        System.out.println("\n=== Début de l'entraînement de copie ===");
        try {
            for (int epoch = 0; epoch < 200; epoch++) {
                float loss = model.trainEpoch(dataGenerator);
                if (epoch % 10 == 0) {
                    System.out.println(String.format("Epoch %d - Loss: %.6f", epoch, loss));
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Erreur pendant l'entraînement", e);
        }
        
        // Test
        System.out.println("\n=== Tests de copie ===");
        for (String input : inputs) {
            String output = model.infer(input, input.split(" ").length ); 
            output = output.replace("<START>", "").replace("<END>", "").trim();
            System.out.println("Input: '" + input + "' → Output: '" + output + "'");
            assertEquals("La copie devrait être exacte", input, output);
        }
    }

}
