package RN.transformer;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

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
    public void testTrainingChangesModelToTrained() throws Exception {
        // Utilisation de DummyDataGenerator pour simuler l'entraînement
        DataGenerator dummyDataGenerator = new DummyDataGenerator("path/to/dummy/data", "path/to/dummy/target", model.tokenizer, 32, 512);
        model.train(dummyDataGenerator);
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());
    }

    @Test(expected = IllegalStateException.class)
    public void testInferenceBeforeTrainingThrowsException() {
        // Tentative d'inférence avant l'entraînement devrait lancer une exception
        model.infer("Some input text");
    }

    @Test
    public void testInferenceAfterTraining() throws Exception {
        // Simuler l'entraînement
        DataGenerator dummyDataGenerator = new DummyDataGenerator("path/to/dummy/data", "path/to/dummy/target", model.tokenizer, 32, 512);
        model.train(dummyDataGenerator);
        
        String response = model.infer("Some input text");
        assertNotNull("L'inférence devrait retourner une réponse non-null", response);
        // Ici, vous pouvez ajouter d'autres assertions pour vérifier la plausibilité de la réponse.
    }
}
