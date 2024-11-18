package RN.transformer;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class TransformerIntegrationTest {

    private TransformerModel model;
    private DataGenerator mockDataGenerator;

    @Before
    public void setUp() throws IOException {
        // Initialisation du Tokenizer avec un vocabulaire simple et dModel = 64
        int maxSequenceLength = 50;
        List<String> vocabulary = Arrays.asList("hello", "world", "test", "input", "output", "<PAD>", "<UNK>", "<START>", "<END>");
        int dModel = 64; // Correspond à la dimension utilisée dans le modèle
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        
        // Initialisation du modèle Transformer avec des paramètres réduits
        int numLayers = 2;
        int numHeads = 4;
        int dff = 255;
        int vocabSize = vocabulary.size();
        float dropoutRate = 0.0f;
        float lr = 0.0001f;
        int warmupSteps = 10;
        
        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, lr, warmupSteps);
        
        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples sans fichiers
        List<String> data = Arrays.asList("hello world");
        List<String> targets = Arrays.asList("hello output");
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, 2, 50);
    } 

    @Test
    public void testTrainingOnMockData() throws Exception {
        // Initialisation de l'entraînement avec un seul epoch
        float loss = model.trainEpochAndGetLoss(mockDataGenerator);

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
        List<String> data = Arrays.asList("hello world", "test input");
        List<String> targets = Arrays.asList("hello output", "test output");
        mockDataGenerator = new MockDataGenerator(data, targets, model.tokenizer, 1, 50, 5); // 5 batches

        // Initialisation des variables pour suivre la perte
        List<Float> lossHistory = new java.util.ArrayList<>();

        // Entraîner sur 5 epochs et enregistrer la perte à chaque epoch
        for (int epoch = 0; epoch < 5; epoch++) {
            float loss = model.trainEpochAndGetLoss(mockDataGenerator);
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
        // Copier les paramètres avant l'entraînement
        List<INDArray> initialParameters = model.getCombinedParameters().stream()
                .map(INDArray::dup)
                .collect(Collectors.toList());

        // Effectuer une étape d'entraînement
        model.trainEpochAndGetLoss(mockDataGenerator);

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
    public void testInferenceAfterTraining() throws Exception {
        // Effectuer l'entraînement
        model.trainEpochAndGetLoss(mockDataGenerator);

        // Effectuer une inférence
        String input = "hello world";
        String actualOutput = model.infer(input, 4);

        // Vérifier que l'inférence est proche de la cible
        // Comme c'est un jeu de données fictif et l'entraînement est limité, cela peut ne pas correspondre exactement
        // Mais vous pouvez vérifier qu'il y a une certaine cohérence ou non-nullité
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Comparer avec une sortie attendue si possible
        String expectedOutput = "hello output";
        //assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    
}
