package RN.transformer;


import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
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

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import RN.utils.NDArrayUtils;

import org.apache.commons.lang3.tuple.Pair;

public class TransformerTest {
    
    private TransformerModel model;

    @Before
    public void setUp() throws Exception {
        // Initialisation de TransformerModel sans lancer d'exception
        model = new TransformerModel( 0.0001f, 10); 
    }
    
    // @After
    // public void tearDown() throws Exception {
    // 	model.cleanGradients();
    //     model = null;
    // }
    
    @Test
    public void testMaskCreation() {
       
        // Créer un batch de tokens sous forme de List<List<Integer>>
        List<List<Integer>> tokensBatch = Arrays.asList(
            Arrays.asList(1, 2, 3, 0, 0) // 0 est supposé être le token de padding
        );
    
        // Convertir le batch de tokens en INDArray
        int batchSize = tokensBatch.size();
        int seqLength = tokensBatch.get(0).size();
        INDArray tokensINDArray = Nd4j.create(batchSize, seqLength);
    
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                tokensINDArray.putScalar(new int[]{i, j}, tokensBatch.get(i).get(j));
            }
        }
        
        // Appeler createPaddingMask avec le batch de tokens
        INDArray paddingMask = NDArrayUtils.createKeyPaddingMask(model.tokenizer, tokensINDArray);
        
        // Afficher le masque généré pour le débogage
        System.out.println("paddingMask: " + paddingMask);
    
        // Créer le masque attendu
        // Avec la nouvelle implémentation, le masque doit être binaire : 1.0f pour les tokens valides, 0.0f pour le padding
        INDArray expectedMask = Nd4j.create(new float[][][][]{
            { { { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f } } }
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
                        assertEquals(expectedValue, actualValue, 1e-5f,
                            String.format("Position (%d,%d,%d,%d) values don't match: expected %f but got %f", i, j, k, l, expectedValue, actualValue));
                    }
                }
            }
        }
        
        // Tester le lookAheadMask si nécessaire
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask(batchSize, 5);
        System.out.println("lookAheadMask: " + lookAheadMask);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expectedValue = j > i ? 0f : 1f;
                float actualValue = lookAheadMask.getFloat(0, 0, i, j);
                assertEquals(expectedValue, actualValue, 1e-5f,
                String.format("Position (%d,%d) values don't match: expected %f but got %f", i, j, expectedValue, actualValue));
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
        Tokenizer tokenizer = new Tokenizer(Arrays.asList(originalText.split(" ")), 300, 11);
        List<String> tokens = tokenizer.tokenize(originalText);
        List<Integer> tokenIds = tokenizer.tokensToIds(tokens);
        String reconstructedText = tokenizer.idsToTokens(tokenIds);
        
        org.junit.Assert.assertEquals("Text should be preserved after tokenization and detokenization", "<START> " + originalText + " <END>", reconstructedText);
    }

    @Test
    public void testEncodeDecode() {
        // Initialiser le Tokenizer avec un vocabulaire connu
        List<String> vocabulary = Arrays.asList("le", "chat", "est", "sur", "tapis", "les", "chiens", "dans", "jardin", "aiment", "manger", "<PAD>", "<UNK>", "<START>", "<END>");
        
        // Initialiser le modèle avec les paramètres appropriés
        int numLayers = 2;
        int dModel = 300;
        int numHeads = 6;
        int dff = 512;
        int vocabSize = vocabulary.size();
        int maxSequenceLength = 50;
        float learningRate = 0.0001f;
        int warmupSteps = 10;
        
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Utiliser le constructeur qui prend un Tokenizer personnalisé
        TransformerModel model = new TransformerModel(numLayers, dModel, numHeads, dff, 0.1, vocabSize, tokenizer, learningRate, warmupSteps);
        
        String testInput = "le chat est sur le tapis les chiens dans le jardin";
        String testExpected = "les chiens aiment le jardin";
        System.out.println("Test input: " + testInput);
        
        List<String> tokensIn = model.tokenizer.tokenize(testInput);
        List<String> tokensOut = model.tokenizer.tokenize(testExpected);
        
        Batch batch = new Batch(tokensIn, tokensOut, model.tokenizer);
        INDArray data = batch.getData();       // [batch_size, seq_length_source]
 
        // Encoder l'entrée
        INDArray encoded = model.encoder.encode(false, batch);
        
        assertNotNull(encoded, "Encoded output should not be null.");
        System.out.println("Encoded output shape: " + Arrays.toString(encoded.shape()));
        
        // Convertir la séquence cible en identifiants d'IDs avec le tokenizer
        List<Integer> targetTokenIds = model.tokenizer.tokensToIds(tokensOut);
        INDArray targetTokenIdsArray = Nd4j.createFromArray(targetTokenIds.stream().mapToInt(i -> i).toArray()).reshape(1, targetTokenIds.size());
    
        // Convertir les IDs de la séquence cible en embeddings
        INDArray encodedDecoderInput = model.tokenizer.lookupEmbeddings(targetTokenIdsArray); // [batch_size, seq_length_target, dModel]
        

        // Décode l'entrée encodée en utilisant les embeddings de la séquence cible
        INDArray decoded = model.decoder.decode(false, encoded, encodedDecoderInput, batch, data);
        assertNotNull(decoded, "Decoded output should not be null.");
        System.out.println("Decoded output shape: " + Arrays.toString(decoded.shape()));
        System.out.println("Decoded output: " + decoded);
        
        // Vérifier la forme de la sortie du décodeur
        assertEquals(1, (int) decoded.shape()[0], "Batch size should be 1");
        
        int outputSize = model.getVocabSize(); // Assurez-vous que cette valeur est correcte
        assertEquals(outputSize, (int) decoded.shape()[2], "Output size should be " + outputSize);
    }
    
    
    @Test
    public void testBackwardPropagation() {
        // Création des entrées fictives
        String testInput = "Test input";
        String testExpected = "output ok";
        List<String> tokensIn = model.tokenizer.tokenize(testInput);
        List<String> tokensOut = model.tokenizer.tokenize(testExpected);
    
        Batch batch = new Batch(tokensIn, tokensOut, model.tokenizer);
        INDArray data = batch.getData();   // Utiliser directement le format INDArray
        INDArray target = batch.getTarget();  // Utiliser directement le format INDArray
        INDArray mask = batch.getMask();  // Masque de padding pour les séquences


        // Passe forward
        INDArray encoded = model.encoder.encode(true, batch);
        INDArray decoderOutput = model.decoder.decode(true, encoded, encoded, batch, data);
    
        // Vérifier la forme des logits
        System.out.println("Decoder Output Shape: " + Arrays.toString(decoderOutput.shape()));
        
        // Création d'un gradient fictif pour la rétropropagation
        INDArray gradOutput = Nd4j.rand(decoderOutput.shape()).castTo(DataType.FLOAT);
    
        // Appel de la méthode backward
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);
    
        // Assertions pour vérifier les gradients
        assertNotNull("Les gradients ne devraient pas être null", gradients);
        assertFalse("Le map des gradients ne devrait pas être vide", gradients.isEmpty());
        
        // Définir les clés attendues
        List<String> expectedKeys = Arrays.asList("input", "gamma", "beta", "W1", "b1", "W2", "b2");
    
        for (String key : expectedKeys) {
            assertTrue("Les gradients devraient contenir la clé '" + key + "'", gradients.containsKey(key));
        }
    
        // S'assurer qu'il n'y a pas de clés inattendues
        // for (String key : gradients.keySet()) {
        //     assertTrue("Clé de gradient inattendue : " + key, expectedKeys.contains(key));
        // }
    
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
                    assertArrayEquals("Forme des gradients de W1", new long[]{model.getDModel(), 2048}, grad.shape());
                    break;
                case "b1":
                    assertArrayEquals("Forme des gradients de b1", new long[]{1, 2048}, grad.shape());
                    break;
                case "W2":
                    assertArrayEquals("Forme des gradients de W2", new long[]{2048, model.getDModel()}, grad.shape());
                    break;
                case "b2":
                    assertArrayEquals("Forme des gradients de b2", new long[]{1, model.getDModel()}, grad.shape());
                    break;
                // case "gamma":
                //     assertArrayEquals("Forme des gradients de gamma", new long[]{1, 512}, grad.shape());
                //     break;
                // case "beta":
                //     assertArrayEquals("Forme des gradients de beta", new long[]{1, 512}, grad.shape());
                //     break;
              
                // default:
                //     fail("Clé de gradient inattendue : " + key);
            }
        }
    }
    

    @Test
    public void testParameterUpdates() throws IOException {
    	
        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples sans fichiers
        List<String> data = Arrays.asList("hello world");
        List<String> targets = Arrays.asList("hello output");
        DataGenerator mockDataGenerator = new DataGenerator(data, targets, model.tokenizer, 1, 50);

        INDArray initialWeights = model.getCombinedParameters().get(5).dup();
        model.train(mockDataGenerator, 1);
        INDArray updatedWeights = model.getCombinedParameters().get(5);
        assertFalse(initialWeights.equalsWithEps(updatedWeights, 1e-6), "Weights should be updated after training.");
    }

    @Test
    public void testLossCalculation() {
        int batchSize = 2;
        int seqLength = 5;
        int vocabSize = 18;
    
        // Assurez-vous que vocabSize est correct
        assertEquals("Vocab size should match", vocabSize, model.getVocabSize());
    
        // Création des logits : [batchSize, seqLength, vocabSize]
        INDArray logitsArray = Nd4j.rand(new int[]{batchSize, seqLength, vocabSize}, 'c');
        List<INDArray> logits = Arrays.asList(logitsArray);
    
        // Création des targetTokenIds : 2 séquences de 5 tokens chacune
        // Assurez-vous que les IDs sont valides (entre 0 et vocabSize -1)
        List<Integer> targetTokenIds1 = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> targetTokenIds2 = Arrays.asList(2, 3, 4, 5, 6);
        List<List<Integer>> targetBatchTokenIds = Arrays.asList(targetTokenIds1, targetTokenIds2);
    
        // Convertir targetBatchTokenIds en INDArray : [batchSize, seqLength]
        INDArray targetBatch = Nd4j.create(batchSize, seqLength);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                targetBatch.putScalar(new int[]{i, j}, targetBatchTokenIds.get(i).get(j));
            }
        }
    
        // Calcul de la perte et des gradients
        Pair<Float, INDArray> lossAndGradients = model.calculateCrossEntropyLossAndGradient(logits, targetBatch);
        float loss = lossAndGradients.getLeft();
        INDArray gradients = lossAndGradients.getRight();
    
        // Vérification que la perte est non négative
        assertTrue("La perte doit être non-négative.", loss >= 0);
    
        // Vérification de la forme des gradients
        assertArrayEquals("Forme des gradients doit correspondre aux logits",
            logitsArray.shape(), gradients.shape());
    
        // Vérification que les gradients sont dans des plages attendues
        assertFalse("Les gradients ne doivent pas contenir de NaN.", gradients.isNaN().any());
        assertFalse("Les gradients ne doivent pas contenir d'Inf.", gradients.isInfinite().any());
    
        // (Optionnel) Vérification de valeurs spécifiques pour des cas simples
        // Vous pouvez définir des logits et des cibles spécifiques et vérifier la perte et les gradients attendus
    }
    


    @Test
    public void testInference() throws IOException {
        // Initialiser le tokenizer et le modèle
        TransformerModel model = new TransformerModel(2, 300, 6, 2048, 0.1, 0.001f, 10);

        // Simuler un entraînement (optionnel)
        model.setTrained(true); // Assurez-vous d'avoir un setter pour cette variable si elle est privée

        // Effectuer l'inférence
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
        TransformerModel loadedModel = new TransformerModel(0.0001f,10);
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
