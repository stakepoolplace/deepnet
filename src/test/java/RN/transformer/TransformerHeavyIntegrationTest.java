// TransformerHeavyIntegrationTest.java
package RN.transformer;

import static org.junit.Assert.*;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class TransformerHeavyIntegrationTest {

    private TransformerModel model;
    private DataGenerator mockDataGenerator;
    private Tokenizer tokenizer;

    @Before
    public void setUp() throws IOException {
        // Initialisation des embeddings pré-entraînés
        WordVectors preTrainedWordVectors = loadPreTrainedWordVectors();

        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length; // Exemple d'obtention de la dimension
        tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize);

        // Initialisation du modèle Transformer avec dModel = embeddingSize
        int numLayers = 2;
        int dModel = embeddingSize;
        int numHeads = 4;
        int dff = 128;
        int vocabSize = tokenizer.getVocabSize();
        float dropoutRate = 0.1f;

        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer);

        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples
        List<String> data = Arrays.asList(
            "chat mange la souris", 
            "chien court dans le jardin", 
            "les chats aiment les chiens",
            "tapis sur le sol"
        );
        List<String> targets = Arrays.asList(
            "le chat mange", 
            "le chien court", 
            "les chats aiment", 
            "le tapis sur"
        );
        int batchSize = 2; // Assurez-vous que cela correspond à la forme des scores
        int sequenceLength = 5; // Définissez une longueur de séquence cohérente
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, sequenceLength);
    }

 

    @Test
    public void testTrainingWithPretrainedEmbeddings() throws Exception {
        
        // Entraîner sur un seul epoch
        float initialLoss = model.trainEpochAndGetLoss(mockDataGenerator);

        // Vérification que le modèle est marqué comme entraîné
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());

        // Effectuer une inférence sur l'entrée d'entraînement
        String input = "chat mange la souris";
        String actualOutput = model.infer(input, 10);

        // Vérification que l'inférence n'est pas nulle et est cohérente
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Vérifier que l'inférence est proche de la cible
        // String expectedOutput = "le chat mange";
        // assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    @Test
    public void testLossDecreaseWithPretrainedEmbeddings() throws Exception {
        // Création d'un DataGenerator avec plusieurs batches pour simuler plusieurs epochs
        List<String> data = Arrays.asList(
            "chat mange la souris", 
            "chien court dans le jardin", 
            "les chats aiment les chiens",
            "tapis sur le sol"
        );
        List<String> targets = Arrays.asList(
            "le chat mange", 
            "le chien court", 
            "les chats aiment", 
            "le tapis sur"
        );
        int batchSize = 1;
        int sequenceLength = 10;
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, sequenceLength);

        // Initialisation des variables pour suivre la perte
        List<Float> lossHistory = new java.util.ArrayList<>();

        // Entraîner sur 5 epochs et enregistrer la perte à chaque epoch
        for (int epoch = 0; epoch < 5; epoch++) {
            float loss = model.trainEpochAndGetLoss(mockDataGenerator);
            lossHistory.add(loss);
            System.out.println("Epoch " + (epoch + 1) + " Loss: " + loss);
            mockDataGenerator.reset(); // Réinitialiser le générateur pour chaque epoch
        }

        // Vérification que la perte diminue au fil des epochs
        for (int i = 1; i < lossHistory.size(); i++) {
            assertTrue("La perte devrait diminuer au fil des epochs",
                    lossHistory.get(i) < lossHistory.get(i - 1));
        }
    }

    @Test
    public void testInferenceWithPretrainedEmbeddings() throws Exception {
        // Effectuer l'entraînement
        float loss = model.trainEpochAndGetLoss(mockDataGenerator);

        // Effectuer une inférence
        String input = "chien court dans le jardin";
        String actualOutput = model.infer(input, 10);

        // Vérifier que l'inférence est proche de la cible
        // String expectedOutput = "le chien court";
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Comparer avec une sortie attendue si possible
        // assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    @Test
    public void testParameterUpdatesWithPretrainedEmbeddings() throws Exception {
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


   /**
     * Charge les WordVectors pré-entraînés.
     * Utilisez WordVectorSerializer pour charger un modèle Word2Vec.
     *
     * @return Instance de WordVectors.
     */
    private WordVectors loadPreTrainedWordVectors() throws IOException {
        // Exemple d'utilisation de WordVectorSerializer pour charger un modèle pré-entraîné
        // Remplacez "path/to/word2vec/model.bin" par le chemin réel de votre modèle
        File modelFile = new File("pretrained-embeddings/mon_model_word2vec.txt");
        if (!modelFile.exists()) {
            throw new IOException("Le fichier du modèle Word2Vec n'existe pas: " + modelFile.getAbsolutePath());
        }
        return org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.readWord2VecModel(modelFile);
    }


}
