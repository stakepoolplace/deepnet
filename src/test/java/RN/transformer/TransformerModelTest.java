package RN.transformer;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TransformerModelTest {

    private TransformerModel model;

    @Before
    public void setUp() throws Exception {
        // Initialisation de TransformerModel sans lancer d'exception
        model = new TransformerModel(); 
    }

    @Test
    public void testModelInitialization() {
        // Vérifications initiales du modèle
        assertNotNull("L'encoder ne devrait pas être null après l'initialisation du modèle", model.encoder);
        assertNotNull("Le decoder ne devrait pas être null après l'initialisation du modèle", model.decoder);
        assertNotNull("L'optimizer ne devrait pas être null après l'initialisation du modèle", model.optimizer);
        assertNotNull("Le tokenizer ne devrait pas être null après l'initialisation du modèle", model.tokenizer);
        assertFalse("Le modèle ne devrait pas être marqué comme entraîné initialement", model.isTrained());
    }
    

    @Test
    public void testSaveAndLoadState() throws IOException, ClassNotFoundException {
        // Créer une instance du modèle
        TransformerModel originalModel = new TransformerModel();

        // Simuler un entraînement en modifiant quelques paramètres
        originalModel.train(new MockDataGenerator(), 1); // Supposons que vous avez une implémentation mock de DataGenerator pour les tests
        
        // Sauvegarder l'état du modèle
        String filePath = "test_transformer_state.ser";
        originalModel.saveState(filePath);
        
        // Créer une nouvelle instance du modèle
        TransformerModel loadedModel = new TransformerModel();
        
        // Charger l'état sauvegardé
        loadedModel.loadState(filePath);
        
        // Vérifier que les états sont identiques
        assertTrue(compareModels(originalModel, loadedModel));
        
        // Nettoyer le fichier de test
        new File(filePath).delete();
    }
    
    private boolean compareModels(TransformerModel model1, TransformerModel model2) {
        // Comparer les paramètres de l'encodeur
        if (!compareParameters(model1.encoder.getParameters(), model2.encoder.getParameters())) {
            return false;
        }
        
        // Comparer les paramètres du décodeur
        if (!compareParameters(model1.decoder.getParameters(), model2.decoder.getParameters())) {
            return false;
        }
        
        // Comparer l'état de l'optimiseur
        if (model1.optimizer.getCurrentStep() != model2.optimizer.getCurrentStep() ||
            model1.optimizer.getEpoch() != model2.optimizer.getEpoch() ||
            model1.optimizer.getLearningRate() != model2.optimizer.getLearningRate()) {
            return false;
        }
        
        // Comparer l'état d'entraînement
        if (model1.isTrained() != model2.isTrained()) {
            return false;
        }
        
        return true;
    }
    
    private boolean compareParameters(List<INDArray> params1, List<INDArray> params2) {
        if (params1.size() != params2.size()) {
            return false;
        }
        
        for (int i = 0; i < params1.size(); i++) {
            if (!params1.get(i).equalsWithEps(params2.get(i), 1e-5)) {
                return false;
            }
        }
        
        return true;
    }
    
    // Classe mock pour DataGenerator
    private class MockDataGenerator extends DataGenerator {
        public MockDataGenerator() throws IOException {
            super("src/test/resources/mock_data.txt", "src/test/resources/mock_target.txt", new Tokenizer(Arrays.asList("tutu", "toto")), 1, 100);
        }
        
        @Override
        public boolean hasNextBatch() {
            return false; // Pour simplifier, on suppose qu'il n'y a pas de batch à traiter
        }
        
        @Override
        public Batch nextBatch() {
            return new Batch(List.of("mock input"), List.of("mock target"));
        }
    }

    @Test
    public void testTrainingChangesModelToTrained() throws Exception {
        // Utilisation de DummyDataGenerator pour simuler l'entraînement
        DataGenerator dummyDataGenerator = new DummyDataGenerator("src/test/resources/dummy-data.txt", "src/test/resources/dummy-data-target.txt", model.tokenizer, 32, 512, 5);
        model.train(dummyDataGenerator, 1);
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());
    }

    @Test(expected = IllegalStateException.class)
    public void testInferenceBeforeTrainingThrowsException() {
        // Tentative d'inférence avant l'entraînement devrait lancer une exception
        model.infer("Some input text", 30);
    }
    


    @Test
    public void testInferenceAfterTraining() throws Exception {
        // Simuler l'entraînement
        DataGenerator dummyDataGenerator = new DummyDataGenerator("src/test/resources/dummy-data.txt", "src/test/resources/dummy-data-target.txt", model.tokenizer, 32, 512, 5);
        model.train(dummyDataGenerator, 1);
        
        String inputPrompt = "Some input text";
        String response = model.infer(inputPrompt, 45);
        assertNotNull("L'inférence devrait retourner une réponse non-null", response);
        System.out.println("Inférence 1 prompt: " + inputPrompt + " : " + response);
        // Ici, vous pouvez ajouter d'autres assertions pour vérifier la plausibilité de la réponse.
        inputPrompt = "This is a dummy sentence";
        response = model.infer(inputPrompt, 32);
        assertNotNull("L'inférence devrait retourner une réponse non-null", response);
        System.out.println("Inférence 2 prompt: " + inputPrompt + " : " + response);

    }

    @Test
    public void testInferenceAfterTraining2() throws Exception {
        // Initialiser le tokenizer et le modèle
        TransformerModel model = new TransformerModel(2, 300, 6, 2048, 0.1); // Utiliser 2 layers pour le test

        // Créer un DummyDataGenerator avec 3 batches par epoch
        DummyDataGenerator dataGenerator = new DummyDataGenerator(
            "src/test/resources/dummy-data.txt",
            "src/test/resources/dummy-data-target.txt",
            model.tokenizer,
            2,  // batchSize
            50, // maxTokensPerBatch
            3   // maxBatchesPerEpoch
        );

        // Simuler l'entraînement
        model.train(dataGenerator, 1);

        // Effectuer l'inférence
        String response1 = model.infer("Some input text", 45);
        assertNotNull("L'inférence devrait retourner une réponse non-null", response1);
        System.out.println("Inference Response 1: " + response1);

        String response2 = model.infer("This is a dummy sentence", 32);
        assertNotNull("L'inférence devrait retourner une réponse non-null", response2);
        System.out.println("Inference Response 2: " + response2);
    }


}
