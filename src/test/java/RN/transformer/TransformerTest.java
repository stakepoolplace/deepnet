package RN.transformer;


import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TransformerTest {
    
    private TransformerModel model;

    @BeforeEach
    public void setUp() throws Exception {
        // Initialize your model and any other components here
        model = new TransformerModel(); // Assuming the constructor is available and initializes everything
    }
    
    @AfterEach
    public void tearDown() throws Exception {
    	model.cleanGradients();
    }
    
    @Test
    public void testMaskCreation() {
        List<Integer> tokens = Arrays.asList(1, 2, 3, 0, 0); // 0 est supposé être le token de padding
        INDArray paddingMask = model.createPaddingMask(tokens);
        INDArray expectedMask = Nd4j.create(new double[]{0, 0, 0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
        
        paddingMask = paddingMask.castTo(expectedMask.dataType());

        // Utiliser equalsWithEps pour comparer les valeurs des tableaux
//        System.out.println(compareWithInfinity(paddingMask,expectedMask, 1e-5));
        
        // Utiliser equalsWithEps pour comparer les valeurs des tableaux
        assertTrue(
            paddingMask.equalsWithEps(expectedMask, 1e-5) || 
            paddingMask.equals(expectedMask),  // Ajout d'une comparaison stricte en cas d'infini
            String.format("Expected mask %s but got %s", expectedMask, paddingMask)
        );
        
        INDArray lookAheadMask = model.createLookAheadMask(5);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double expectedValue = j > i ? Double.POSITIVE_INFINITY : 0.0;
                double actualValue = lookAheadMask.getDouble(i, j);
                if (Double.isInfinite(expectedValue)) {
                    assertTrue(Double.isInfinite(actualValue),
                            String.format("Position (%d,%d) expected Infinite but got %f", i, j, actualValue));
                } else {
                    assertEquals(expectedValue, actualValue, 1e-5,
                            String.format("Position (%d,%d) values don't match", i, j));
                }
            }
        }
    }
    
    public static boolean compareWithInfinity(INDArray matrix1, INDArray matrix2, double epsilon) {
        
    	// Vérifier si les deux matrices ont la même forme
        if (!matrix1.shapeInfoToString().equals(matrix2.shapeInfoToString())) {
            return false;
        }

        // Parcourir chaque élément pour comparer
        for (int i = 0; i < matrix1.length(); i++) {
            double val1 = matrix1.getDouble(i);
            double val2 = matrix2.getDouble(i);

            if (Double.isInfinite(val1) && Double.isInfinite(val2)) {
                // Si les deux valeurs sont infinies, elles sont considérées égales
                continue;
            }

            if (Double.isNaN(val1) || Double.isNaN(val2)) {
                // Gérer les NaN explicitement
                if (Double.isNaN(val1) != Double.isNaN(val2)) {
                    return false;
                }
            } else if (Math.abs(val1 - val2) > epsilon) {
                // Comparaison normale avec tolérance epsilon
                return false;
            }
        }

        return true;
    }
    
    
    @Test
    public void testTokenizationAndDetokenization() {
        String originalText = "The quick brown fox jumps over the lazy dog";
        Tokenizer tokenizer = new Tokenizer(Arrays.asList(originalText.split(" ")));
        List<String> tokens = tokenizer.tokenize(originalText);
        List<Integer> tokenIds = tokenizer.tokensToIds(tokens);
        String reconstructedText = tokenizer.idsToTokens(tokenIds);
        
        org.junit.Assert.assertEquals(originalText, reconstructedText, "Text should be preserved after tokenization and detokenization");
    }

    @Test
    public void testEncodeDecode() {
        // Initialiser le Tokenizer avec un vocabulaire de mots connus
//        Collection<String> vocabulary = List.of("le", "chat", "est", "sur", "tapis", "les", "chiens", "dans", "jardin", "aiment", "manger");
//        model.tokenizer = new Tokenizer(vocabulary);

        String testInput = "le chat est sur le tapis les chiens dans le jardin";
        System.out.println("Test input: " + testInput);
        List<String> tokens = model.tokenizer.tokenize(testInput);
        System.out.println("Tokens: " + tokens);
        
        // Vérification du vocabulaire
        int vocabSize = TransformerModel.getVocabSize();
        System.out.println("Tokenizer Vocabulary Size: " + vocabSize);
        tokens.forEach(token -> {
            Integer tokenId = model.tokenizer.tokensToIds(List.of(token)).get(0);
            System.out.println("Token: \"" + token + "\" -> Token ID: " + tokenId);
        });
        
        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        System.out.println("Input Token IDs: " + inputTokenIds);
        
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokens);

        
        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIds);
        INDArray decoderPaddingMask = model.createPaddingMask(targetTokenIds);
        INDArray lookAheadMask = model.createLookAheadMask(targetTokenIds.size());
        
        
        System.out.println("Encoder Padding Mask shape: " + Arrays.toString(encoderPaddingMask.shape()));
        System.out.println("Encoder Padding Mask: " + encoderPaddingMask);
        
        // Encoder l'entrée
        INDArray encoded = model.encoder.encode(false, inputTokenIds, encoderPaddingMask);
        assertNotNull(encoded, "Encoded output should not be null.");
        System.out.println("Encoded output shape: " + Arrays.toString(encoded.shape()));

        // Décoder la sortie
        INDArray decoded = model.decoder.decode(false, encoded, encoded, lookAheadMask, decoderPaddingMask);
        assertNotNull(decoded, "Decoded output should not be null.");
        System.out.println("Decoded output shape: " + Arrays.toString(decoded.shape()));
        System.out.println("Decoded output: " + decoded);
        
        assertEquals(model.tokenizer.getVocabSize(), decoded.shape()[1], "Decoded output should have logits for each token in vocabulary");
    }

    @Test
    public void testBackwardPropagation() {
        // D'abord, effectuez un passage avant pour initialiser les caches
        String testInput = "Test input";
        List<String> tokens = model.tokenizer.tokenize(testInput);

        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokens);

        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIds);
        INDArray decoderPaddingMask = model.createPaddingMask(targetTokenIds);
        INDArray lookAheadMask = model.createLookAheadMask(targetTokenIds.size());

        INDArray encoded = model.encoder.encode(true, inputTokenIds, encoderPaddingMask);
        INDArray decoderOutput = model.decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);
        
        // Maintenant, effectuez la rétropropagation
        INDArray gradOutput = Nd4j.rand(decoderOutput.shape());
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);
        assertNotNull(gradients, "Gradients should not be null.");
        assertFalse(gradients.isEmpty(), "Gradients map should not be empty.");
    }
    
    @Test
    public void testBackwardPropagation1() {
        // Initialisation des paramètres du modèle
        int numLayers = 6;
        int dModel = 512;
        int numHeads = 8;
        int dff = 2048;
        double dropoutRate = 0.1;
        TransformerModel model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate); // Assurez-vous que TransformerModel est correctement initialisé

        // Création des entrées fictives
        String testInput = "Test input";
        List<String> tokens = model.tokenizer.tokenize(testInput);

        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokens);

        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIds);
        INDArray decoderPaddingMask = model.createPaddingMask(targetTokenIds);
        INDArray lookAheadMask = model.createLookAheadMask(targetTokenIds.size());

        // Passe forward
        INDArray encoded = model.encoder.encode(true, inputTokenIds, encoderPaddingMask);
        INDArray decoderOutput = model.decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);

        // Création d'un gradient fictif pour la rétropropagation
        INDArray gradOutput = Nd4j.rand(decoderOutput.shape());

        // Appel de la méthode backward
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);

        // Assertions pour vérifier les gradients
        assertNotNull("Les gradients ne devraient pas être null", gradients);
        assertFalse("Le map des gradients ne devrait pas être vide", gradients.isEmpty());
        assertTrue("Les gradients devraient contenir la clé 'gamma'", gradients.containsKey("gamma"));
        assertTrue("Les gradients devraient contenir la clé 'beta'", gradients.containsKey("beta"));
//        assertTrue("Les gradients devraient contenir la clé 'weights'", gradients.containsKey("weights"));
//        assertTrue("Les gradients devraient contenir la clé 'bias'", gradients.containsKey("bias"));

        // Optionnel : Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            INDArray grad = entry.getValue();
            assertFalse("Le gradient pour " + entry.getKey() + " contient des NaN", grad.isNaN().any());
            assertFalse("Le gradient pour " + entry.getKey() + " contient des Inf", grad.isInfinite().any());
        }
    }

    @Test
    public void testParameterUpdates() throws IOException {
    	
        // Create temporary test files
        File tempFile = File.createTempFile("test", ".tmp");
        tempFile.deleteOnExit();
        
        // Créez un DataGenerator mock qui ne nécessite pas de fichiers réels
        DataGenerator mockDataGenerator = new DataGenerator(tempFile.getPath(), tempFile.getPath(), model.tokenizer, 1, 256);
        
        model.addCombinedParameters();
        INDArray initialWeights = model.getCombinedParameters().get(0).dup();
        model.train(mockDataGenerator);
        INDArray updatedWeights = model.getCombinedParameters().get(0);
        assertFalse(initialWeights.equalsWithEps(updatedWeights, 1e-6), "Weights should be updated after training.");
    }

    @Test
    public void testLossCalculation() {
        int batchSize = 2;
        int seqLength = 5;

        // Création des logits : batchSize x seqLength x vocabSize
        List<INDArray> logits = Arrays.asList(Nd4j.rand(new int[]{batchSize * seqLength, TransformerModel.getVocabSize()}));

        // Création des targetTokenIds : 2 séquences concaténées de 5 tokens (10 tokens au total)
        List<Integer> targetTokenIds = Arrays.asList(0, 1, 2, 3, 4, 0, 1, 2, 3, 4);

        // Calcul de la perte
        float loss = model.calculateCrossEntropyLossAndGradient(logits, targetTokenIds).getLeft();

        // Vérification que la perte est non négative
        assertTrue(loss >= 0, "La perte doit être non-négative.");
    }



    @Test
    public void testInference() {
        // D'abord, simulez un entraînement pour que le modèle soit considéré comme entraîné
        model.setTrained(true); // Assurez-vous d'avoir un setter pour cette variable si elle est privée
        
        String response = model.infer("Hello world", 100);
        assertFalse(response.isEmpty(), "Response should not be empty.");
    }
    
    @Test
    public void testTrainingState() {
        assertFalse(model.isTrained(), "Model should not be trained initially");
        
        // Simulez un entraînement
        model.setTrained(true); // Assurez-vous d'avoir un setter pour cette variable si elle est privée
        
        assertTrue(model.isTrained(), "Model should be marked as trained after training");
    }

    @Test
    public void testSaveAndLoadState() throws IOException, ClassNotFoundException {
        // Simuler un entraînement
        model.setTrained(true); // Assurez-vous d'avoir un setter pour cette variable si elle est privée
        INDArray initialWeights = model.encoder.getParameters().get(0).dup();
        
        // Sauvegarder l'état
        String filePath = "test_model_state.ser";
        model.saveState(filePath);
        
        // Créer un nouveau modèle et charger l'état
        TransformerModel loadedModel = new TransformerModel();
        loadedModel.loadState(filePath);
        
        // Vérifier que l'état chargé correspond à l'état sauvegardé
        assertTrue(loadedModel.isTrained(), "Loaded model should be marked as trained");
        INDArray loadedWeights = loadedModel.encoder.getParameters().get(0);
        assertTrue(initialWeights.equalsWithEps(loadedWeights, 1e-6), "Loaded weights should match initial weights");
        
        // Nettoyer le fichier de test
        new File(filePath).delete();
    }    
    
    @Test
    public void testOptimizerUpdate() {
    	model.addCombinedParameters();
        List<INDArray> parameters = model.getCombinedParameters();
        List<INDArray> gradients = new ArrayList<>();
        for (INDArray param : parameters) {
            gradients.add(Nd4j.rand(param.shape()));
        }
        
        // Copier les paramètres initiaux
        List<INDArray> initialParams = new ArrayList<>();
        for (INDArray param : parameters) {
            initialParams.add(param.dup());
        }
        
        // Effectuer une mise à jour
        model.optimizer.update(parameters, gradients);
        
        // Vérifier que les paramètres ont été mis à jour
        for (int i = 0; i < parameters.size(); i++) {
            assertFalse(parameters.get(i).equalsWithEps(initialParams.get(i), 1e-6), 
                        "Parameter " + i + " should have been updated");
        }
    }
    
    @Test
    public void testAdaptiveLearningRate() {
        double initialLr = model.optimizer.getLearningRate();
        
        // Simuler plusieurs étapes d'entraînement
        for (int i = 0; i < 2000; i++) {
            model.optimizer.setCurrentStep(i);
            model.optimizer.calculateLearningRate();

        }
        
        double laterLr = model.optimizer.getLearningRate();
        assertNotEquals(initialLr, laterLr, "Learning rate should change over time");
    }
    
    
    
//    @Test
//    public void testLossDecrease() {
//        // Assuming you have a method to run multiple epochs and return the last loss
//        double initialLoss = model.runEpochAndGetLoss();
//        double laterLoss = model.runEpochAndGetLoss();
//        assertTrue(laterLoss < initialLoss, "Loss should decrease after training for an epoch.");
//    }
//
//    @Test
//    public void testGradientsNonZero() {
//        // Run a single backward step and check gradients
//        model.train(new DataGenerator("dummy_path", "dummy_target", model.tokenizer, 1, 10));
//        boolean allNonZero = model.decoder.getGradients().stream()
//                             .allMatch(g -> !g.isZeroNumber());
//        assertTrue(allNonZero, "All gradients should be non-zero after training step.");
//    }
//
//    @Test
//    public void testOutputRange() {
//        // Assuming outputs are probabilities from the last layer
//        double[] outputs = model.forwardPassAndGetOutputs(new double[]{0.1, 0.2, 0.7}); // Example input
//        for (double output : outputs) {
//            assertTrue(output >= 0 && output <= 1, "Each output should be a valid probability.");
//        }
//        assertEquals(1.0, Arrays.stream(outputs).sum(), 0.01, "Sum of output probabilities should be 1.");
//    }
}
