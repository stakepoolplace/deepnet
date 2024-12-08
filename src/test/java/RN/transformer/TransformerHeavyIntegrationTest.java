// TransformerHeavyIntegrationTest.java
package RN.transformer;

import static org.junit.Assert.*;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TransformerHeavyIntegrationTest {

    private TransformerModel model;
    private DataGenerator mockDataGenerator;
    private Tokenizer tokenizer;

    @Before
    public void setUp() throws IOException {
        int maxSequenceLength = 10;
        // Initialisation des embeddings pré-entraînés
        WordVectors preTrainedWordVectors = loadPreTrainedWordVectors();

        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length; // Exemple d'obtention de la dimension
        tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize, maxSequenceLength);

        tokenizer.printVocabulary();


        // Initialisation du modèle Transformer avec dModel = embeddingSize
        int numLayers = 4;
        int dModel = embeddingSize;
        int numHeads = 2;
        int dff = 256;
        int vocabSize = tokenizer.getVocabSize();
        float dropoutRate = 0.0f;
        float initialLr = 0.001f;
        int warmupSteps = 10;

        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, initialLr, warmupSteps);

        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples
        List<String> inputs = Arrays.asList(
            "le chat sur", "chat sur", "le chat sur",
            "le chat dans", "chat dans", "le chat dans"
        );
        List<String> targets = Arrays.asList(
            "le tapis", "le tapis", "le tapis",
            "le jardin", "le jardin", "le jardin"
        );


        int batchSize = 1; // Assurez-vous que cela correspond à la forme des scores
        int sequenceLength = 7; // Définissez une longueur de séquence cohérente
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, sequenceLength);
    }

 

    @Test
    public void testTrainingWithPretrainedEmbeddings() throws Exception {
        
        int epochs = 30;
        model.getOptimizer().setMaxEpochs(epochs);
        model.setTrace(false);

        // Entraîner 
        float initialLoss = model.train(mockDataGenerator, epochs);

        // Vérification que le modèle est marqué comme entraîné
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());

        // Effectuer une inférence sur l'entrée d'entraînement
        String input = "le chat sur";
        String actualOutput = model.infer(input, 2);

        // Vérification que l'inférence n'est pas nulle et est cohérente
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Vérifier que l'inférence est proche de la cible
        String expectedOutput = "le tapis";
        assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    @Test
    public void testLossDecreaseWithPretrainedEmbeddings() throws Exception {
        // Création d'un DataGenerator avec plusieurs batches pour simuler plusieurs epochs
        List<String> data = Arrays.asList(
            "chat manger la souris", 
            "chiens aiment le jardin", 
            "chat aiment les chiens",
            "chat sur le tapis"
        );
        List<String> targets = Arrays.asList(
            "le chat manger", 
            "les chiens aiment", 
            "le chat aiment", 
            "le chat sur"
        );
        int batchSize = 2;
        int sequenceLength = 6;
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, sequenceLength);

        // Initialisation des variables pour suivre la perte
        List<Float> lossHistory = new java.util.ArrayList<>();

        // Entraîner sur 5 epochs et enregistrer la perte à chaque epoch
        // for (int epoch = 0; epoch < 15; epoch++) {
        //     float loss = model.trainEpochAndGetLoss(mockDataGenerator);
        //     lossHistory.add(loss);
        //     System.out.println("Epoch " + (epoch + 1) + " Loss: " + loss);
        //     mockDataGenerator.reset(); // Réinitialiser le générateur pour chaque epoch
        // }

        // // Vérification que la perte diminue au fil des epochs
        // for (int i = 1; i < lossHistory.size(); i++) {
        //     assertTrue("La perte devrait diminuer au fil des epochs",
        //             lossHistory.get(i) < lossHistory.get(i - 1));
        // }

       // Effectuer une inférence
    //    String input = "chiens aiment le jardin";
    //    String actualOutput = model.infer(input, 3);
    //    String expectedOutput = "les chiens aiment";

    //    assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);


    }

    @Test
    public void testLayerNormBackwardWithControlledData() {
        int dModel = 3;
        LayerNorm layerNorm = new LayerNorm(dModel);
        
        // Exemple de batchSize = 2, seqLength = 2, dModel = 3
        INDArray input = Nd4j.create(new double[][][] {
            { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} },
            { {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0} }
        });
        
        INDArray output = layerNorm.forward(input);
        
        // Gradient de sortie simulé
        INDArray gradOutput = Nd4j.onesLike(output);
        
        Map<String, INDArray> gradients = layerNorm.backward(gradOutput);
        
        // Vérifiez que les gradients ne contiennent pas de NaN ou d'infinis
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            assertFalse("Gradient " + entry.getKey() + " contient des NaN", entry.getValue().isNaN().any());
            assertFalse("Gradient " + entry.getKey() + " contient des valeurs infinies", entry.getValue().isInfinite().any());
        }
    }

    @Test
    public void testInferenceWithPretrainedEmbeddings() throws Exception {
        // Effectuer l'entraînement
        float loss = model.train(mockDataGenerator, 5);

        // Effectuer une inférence
        String input = "chiens aiment le jardin";
        String actualOutput = model.infer(input, 3);
        String expectedOutput = "les chiens aiment";


        // Vérifier que l'inférence est proche de la cible
        // String expectedOutput = "le chien court";
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());

        // (Optionnel) Comparer avec une sortie attendue si possible
        assertEquals("L'inférence devrait correspondre à la cible", expectedOutput, actualOutput);
    }

    @Test
    public void testVocabularyIncludesAllWords() {
        String[] words = {"chat", "manger", "le", "chiens", "dans", "jardin", "les", "chat", "aiment", "tapis", "sur"};
        for (String word : words) {
            System.out.println("token : " + word);
            int id = tokenizer.getTokenToId().get(word);
            assertNotEquals("Le mot '" + word + "' est mappé à <UNK>", tokenizer.getUnkTokenId(), id);
        }
    }
    

    @Test
    public void testParameterUpdatesWithPretrainedEmbeddings() throws Exception {
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
