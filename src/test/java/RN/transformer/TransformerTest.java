package RN.transformer;


import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Before;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.commons.lang3.tuple.Pair;

public class TransformerTest {
    
    private TransformerModel model;

    private int dModel = 10;
    private int vocabSize = 15;


    @BeforeEach
    public void setUp() throws Exception {
        // Initialize your model and any other components here
        model = new TransformerModel(6, this.dModel, 2, 1200, 0.1);
    }  
    
    @AfterEach
    public void tearDown() throws Exception {
    	model.cleanGradients();
    }
    
    @Test
    public void testMaskCreation() {
        // Créer un batch de tokens (List<List<Integer>>)
        List<List<Integer>> tokensBatch = Arrays.asList(
            Arrays.asList(1, 2, 3, 0, 0) // 0 est supposé être le token de padding
        );
        
        // Appeler createPaddingMask avec le batch de tokens
        INDArray paddingMask = model.createPaddingMask(tokensBatch);
        
        // Afficher le masque généré pour le débogage
        System.out.println("paddingMask: " + paddingMask);
        
        // Créer le masque attendu
        // Comme le masque a la forme [batchSize, 1, 1, seqLength], nous devons créer un tableau 4D
        INDArray expectedMask = Nd4j.create(new float[][][][]{
            { { { 0.0f, 0.0f, 0.0f, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY } } }
        });
        
        // S'assurer que les types de données correspondent
        paddingMask = paddingMask.castTo(expectedMask.dataType());
        
        // Comparer les masques élément par élément
        long[] shape = paddingMask.shape();
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        float actualValue = paddingMask.getFloat(i, j, k, l);
                        float expectedValue = expectedMask.getFloat(i, j, k, l);
                        if (Float.isInfinite(expectedValue)) {
                            assertTrue(Float.isInfinite(actualValue),
                                    String.format("Position (%d,%d,%d,%d) expected Infinite but got %f", i, j, k, l, actualValue));
                        } else {
                            assertEquals(expectedValue, actualValue, 1e-5f,
                                    String.format("Position (%d,%d,%d,%d) values don't match", i, j, k, l));
                        }
                    }
                }
            }
        }
        
        // Tester le lookAheadMask si nécessaire
        INDArray lookAheadMask = model.createLookAheadMask(5);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expectedValue = j > i ? Float.NEGATIVE_INFINITY : 0.0f;
                float actualValue = lookAheadMask.getFloat(i, j);
                if (Float.isInfinite(expectedValue)) {
                    assertTrue(Float.isInfinite(actualValue),
                            String.format("Position (%d,%d) expected Infinite but got %f", i, j, actualValue));
                } else {
                    assertEquals(expectedValue, actualValue, 1e-5f,
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
        
        org.junit.Assert.assertEquals("Text should be preserved after tokenization and detokenization", originalText, reconstructedText);
    }

    @Test
    public void testEncodeDecode() {
        // Initialize the Tokenizer with a known vocabulary
        List<String> vocabulary = Arrays.asList("le", "chat", "est", "sur", "tapis", "les", "chiens", "dans", "jardin", "aiment", "manger");
        Tokenizer tokenizer = new Tokenizer(vocabulary);
    
        // Initialize the model with appropriate parameters
        int numLayers = 2;
        int dModel = 300;
        int numHeads = 6;
        int dff = 512;
        int vocabSize = vocabulary.size();
        TransformerModel model = new TransformerModel(numLayers, dModel, numHeads, dff, vocabSize);
        model.tokenizer = tokenizer; // Associate the tokenizer with the model
    
        String testInput = "le chat est sur le tapis les chiens dans le jardin";
        System.out.println("Test input: " + testInput);
        List<String> tokens = model.tokenizer.tokenize(testInput);
        System.out.println("Tokens: " + tokens);
    
        // Token IDs
        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        System.out.println("Input Token IDs: " + inputTokenIds);
    
        // Prepare inputTokenIdsBatch as List<List<Integer>>
        List<List<Integer>> inputTokenIdsBatch = Arrays.asList(inputTokenIds);
    
        // Convert inputTokenIdsBatch to INDArray for encoder input
        int batchSize = inputTokenIdsBatch.size();
        int seqLength = inputTokenIds.size();
    
        // Create an INDArray for encoder input
        INDArray encoderInput = Nd4j.create(DataType.INT32, batchSize, seqLength);
        for (int i = 0; i < batchSize; i++) {
            List<Integer> sequence = inputTokenIdsBatch.get(i);
            for (int j = 0; j < sequence.size(); j++) {
                encoderInput.putScalar(new int[]{i, j}, sequence.get(j));
            }
        }
    
        // Create padding masks
        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIdsBatch);
        INDArray decoderPaddingMask = model.createPaddingMask(inputTokenIdsBatch);
        INDArray lookAheadMask = model.createLookAheadMask(seqLength);
    
        System.out.println("Encoder Padding Mask shape: " + Arrays.toString(encoderPaddingMask.shape()));
        System.out.println("Encoder Padding Mask: " + encoderPaddingMask);
    

        List<List<Integer>> inputTokenIdsBatchFromArray = new ArrayList<>();
        List<Integer> sequence = new ArrayList<>();
        for (int j = 0; j < seqLength; j++) {
            sequence.add(encoderInput.getInt(0, j));
        }
        inputTokenIdsBatchFromArray.add(sequence);

        // Encode the input
        INDArray encoded = model.encoder.encode(false, inputTokenIdsBatchFromArray, encoderPaddingMask);

        assertNotNull(encoded, "Encoded output should not be null.");
        System.out.println("Encoded output shape: " + Arrays.toString(encoded.shape()));
    
        // Verify the shape of the encoding
        assertEquals(1, (int) encoded.shape()[0], "Batch size should be 1");
        assertEquals(seqLength, (int) encoded.shape()[1], "Sequence length should match the input tokens size");
        assertEquals(dModel, (int) encoded.shape()[2], "dModel should be " + dModel);
    
        // Préparer decoderInput comme un tenseur de rang 3 et de type FLOAT
        INDArray decoderInput = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);
        System.out.println("decoderInput shape: " + Arrays.toString(decoderInput.shape()));

        // Decode the output
        INDArray decoded = model.decoder.decode(false, decoderInput, encoded, lookAheadMask, decoderPaddingMask);
        assertNotNull(decoded, "Decoded output should not be null.");
        System.out.println("Decoded output shape: " + Arrays.toString(decoded.shape()));
        System.out.println("Decoded output: " + decoded);
    
        // Verify the shape of the decoder output
        assertEquals(1, (int) decoded.shape()[0], "Batch size should be 1");
        assertEquals(seqLength, (int) decoded.shape()[1], "Sequence length should match the input tokens size");

        int outputSize = TransformerModel.getVocabSize(); // Assurez-vous que cette valeur est correcte
        assertEquals(outputSize, (int) decoded.shape()[2], "Output size should be " + outputSize);

    }
    
        
    @Test
    public void testBackwardPropagation0() {
        // Initialiser les données d'entrée
        String testInput = "Test input";
        List<String> tokens = model.tokenizer.tokenize(testInput);

        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokens);

       // Prepare inputTokenIdsBatch as List<List<Integer>>
       List<List<Integer>> inputTokenIdsBatch = Arrays.asList(inputTokenIds);
       List<List<Integer>> targetTokenIdsBatch = Arrays.asList(targetTokenIds);
    

        // Créer les masques
        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIdsBatch);
        INDArray decoderPaddingMask = model.createPaddingMask(targetTokenIdsBatch);
        INDArray lookAheadMask = model.createLookAheadMask(targetTokenIds.size());

        // Effectuer le passage avant
        INDArray encoded = model.encoder.encode(true, inputTokenIdsBatch, encoderPaddingMask);
        INDArray decoderOutput = model.decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);

        // Vérifier la forme de la sortie
        Assert.assertEquals("Decoder output should have rank 3", 3, decoderOutput.rank());
        Assert.assertEquals("Last dimension of decoder output should be vocabSize", 
                            vocabSize, (int) decoderOutput.shape()[2]);

        // Effectuer la rétropropagation avec un gradOutput aléatoire
        INDArray gradOutput = Nd4j.rand(decoderOutput.shape());
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);

        // Vérifier que les gradients ne sont pas nuls ou vides
        Assert.assertNotNull("Gradients should not be null.", gradients);
        Assert.assertFalse("Gradients map should not be empty.", gradients.isEmpty());

        // Vérifier que les gradients contiennent les clés attendues pour LinearProjection
        Assert.assertTrue("Gradients should contain 'weights'", gradients.containsKey("weights"));
        Assert.assertTrue("Gradients should contain 'bias'", gradients.containsKey("bias"));
        Assert.assertTrue("Gradients should contain 'gamma'", gradients.containsKey("gamma"));
        Assert.assertTrue("Gradients should contain 'beta'", gradients.containsKey("beta"));

        // Vérifier les formes des gradients de LinearProjection
        INDArray gradWeights = gradients.get("weights");
        INDArray gradBias = gradients.get("bias");
        INDArray gradGamma = gradients.get("gamma");
        INDArray gradBeta = gradients.get("beta");

        Assert.assertArrayEquals("weights gradient shape should be [dModel, vocabSize]", 
                                  new long[]{dModel, vocabSize}, gradWeights.shape());
        Assert.assertArrayEquals("bias gradient shape should be [1, vocabSize]", 
                                  new long[]{1, vocabSize}, gradBias.shape());
        Assert.assertArrayEquals("gamma gradient shape should be [1, dModel]", 
                                  new long[]{1, dModel}, gradGamma.shape());
        Assert.assertArrayEquals("beta gradient shape should be [1, dModel]", 
                                  new long[]{1, dModel}, gradBeta.shape());

    }
    
    @Test
    public void testBackwardPropagation1() {
        // Création des entrées fictives
        String testInput = "Test input";
        List<String> tokens = model.tokenizer.tokenize(testInput);

        List<Integer> inputTokenIds = model.tokenizer.tokensToIds(tokens);
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokens);

        // Préparer inputTokenIdsBatch et targetTokenIdsBatch comme List<List<Integer>>
        List<List<Integer>> inputTokenIdsBatch = Arrays.asList(inputTokenIds);
        List<List<Integer>> targetTokenIdsBatch = Arrays.asList(targetTokenIds);

        INDArray encoderPaddingMask = model.createPaddingMask(inputTokenIdsBatch);
        INDArray decoderPaddingMask = model.createPaddingMask(targetTokenIdsBatch);
        INDArray lookAheadMask = model.createLookAheadMask(targetTokenIds.size());

        // Passe forward
        INDArray encoded = model.encoder.encode(true, inputTokenIdsBatch, encoderPaddingMask);
        INDArray decoderOutput = model.decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);

        // Vérifier la forme des logits
        System.out.println("Decoder Output Shape: " + Arrays.toString(decoderOutput.shape()));
        
        // Création d'un gradient fictif pour la rétropropagation
        INDArray gradOutput = Nd4j.rand(decoderOutput.shape()).castTo(DataType.FLOAT);

        // Appel de la méthode backward
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);

        // Assertions pour vérifier les gradients
        assertNotNull("Les gradients ne devraient pas être null", gradients);
        assertFalse("Le map des gradients ne devrait pas être vide", gradients.isEmpty());
        // Décommentez si vous souhaitez vérifier 'gamma' et 'beta' si ils sont présents
        assertTrue("Les gradients devraient contenir la clé 'gamma'", gradients.containsKey("gamma"));
        assertTrue("Les gradients devraient contenir la clé 'beta'", gradients.containsKey("beta"));
        // Décommentez si vous souhaitez vérifier 'W1', 'b1', 'W2', 'b2', etc.
        // assertTrue("Les gradients devraient contenir la clé 'W1'", gradients.containsKey("W1"));
        // assertTrue("Les gradients devraient contenir la clé 'b1'", gradients.containsKey("b1"));
        // assertTrue("Les gradients devraient contenir la clé 'W2'", gradients.containsKey("W2"));
        // assertTrue("Les gradients devraient contenir la clé 'b2'", gradients.containsKey("b2"));

        // Optionnel : Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            INDArray grad = entry.getValue();
            assertFalse("Le gradient pour " + entry.getKey() + " contient des NaN", grad.isNaN().any());
            assertFalse("Le gradient pour " + entry.getKey() + " contient des Inf", grad.isInfinite().any());
        }

        // Vérifier que les gradients ont des formes cohérentes
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            String key = entry.getKey();
            INDArray grad = entry.getValue();
            switch (key) {
                case "W1":
                    assertEquals("Forme des gradients de W1", new long[]{512, 2048}, grad.shape()); // ffSize = 2048
                    break;
                case "b1":
                    assertEquals("Forme des gradients de b1", new long[]{1, 2048}, grad.shape());
                    break;
                case "W2":
                    assertEquals("Forme des gradients de W2", new long[]{2048, 512}, grad.shape());
                    break;
                case "b2":
                    assertEquals("Forme des gradients de b2", new long[]{1, 512}, grad.shape());
                    break;
                case "input":
                    // Forme du gradient de l'entrée dépend de l'entrée originale
                    assertEquals("Forme du gradient de l'entrée", encoded.shape(), grad.shape());
                    break;
                case "gamma":
                    assertEquals("Forme des gradients de gamma", new long[]{1, 512}, grad.shape());
                    break;
                case "beta":
                    assertEquals("Forme des gradients de beta", new long[]{1, 512}, grad.shape());
                    break;
                default:
                    fail("Clé de gradient inattendue : " + key);
            }
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

        // Assurez-vous que vocabSize est correct
        assertEquals("Vocab size should match", vocabSize, TransformerModel.getVocabSize());

        // Création des logits : [batchSize, seqLength, vocabSize]
        INDArray logitsArray = Nd4j.rand(new int[]{batchSize, seqLength, vocabSize}, 'c');
        List<INDArray> logits = Arrays.asList(logitsArray);

        // Création des targetTokenIds : 2 séquences de 5 tokens chacune
        // Assurez-vous que les IDs sont valides (entre 0 et vocabSize -1)
        List<Integer> targetTokenIds1 = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> targetTokenIds2 = Arrays.asList(2, 3, 4, 5, 6);
        List<List<Integer>> targetBatchTokenIds = Arrays.asList(targetTokenIds1, targetTokenIds2);

        // Calcul de la perte et des gradients
        Pair<Float, INDArray> lossAndGradients = model.calculateCrossEntropyLossAndGradient(logits, targetBatchTokenIds);
        float loss = lossAndGradients.getLeft();
        INDArray gradients = lossAndGradients.getRight();

        // Vérification que la perte est non négative
        assertTrue("La perte doit être non-négative.", loss >= 0);

        // Vérification de la forme des gradients
        assertArrayEquals("Forme des gradients doit correspondre aux logits",
            logitsArray.shape(), gradients.shape());

        // // Vérification que les gradients sont dans des plages attendues
        // assertFalse("Les gradients ne doivent pas contenir de NaN.", gradients.isNaN());
        // assertFalse("Les gradients ne doivent pas contenir d'Inf.", gradients.isInfinite());

        // (Optionnel) Vérification de valeurs spécifiques pour des cas simples
        // Vous pouvez définir des logits et des cibles spécifiques et vérifier la perte et les gradients attendus
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
        // Ajouter les paramètres au modèle
        model.addCombinedParameters();
        
        // Récupérer les paramètres ajoutés
        List<INDArray> parameters = model.getCombinedParameters();
        
        // Initialiser l'optimiseur **après** l'ajout des paramètres
        float initialLr = 0.001f;
        int dmodel = 3;
        int warmupSteps = 1000;
        model.optimizer = new CustomAdamOptimizer(initialLr, dmodel, warmupSteps, parameters);
        
        // Générer des gradients fictifs
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
    public void testOptimizerMultipleUpdates() {
        // Ajouter les paramètres au modèle
        model.addCombinedParameters();
        
        // Récupérer les paramètres ajoutés
        List<INDArray> parameters = model.getCombinedParameters();
        
        // Initialiser l'optimiseur avec warmupSteps = 1
        float initialLr = 0.001f;
        int dmodel = 3;
        int warmupSteps = 1;
        model.optimizer = new CustomAdamOptimizer(initialLr, dmodel, warmupSteps, parameters);
        
        // Générer des gradients fictifs
        List<INDArray> gradients = new ArrayList<>();
        for (INDArray param : parameters) {
            gradients.add(Nd4j.rand(param.shape()));
        }
        
        // Copier les paramètres initiaux
        List<INDArray> initialParams = new ArrayList<>();
        for (INDArray param : parameters) {
            initialParams.add(param.dup());
        }
        
        // Effectuer plusieurs mises à jour
        int numUpdates = 10;
        for (int u = 0; u < numUpdates; u++) {
            model.optimizer.update(parameters, gradients);
        }
        
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
