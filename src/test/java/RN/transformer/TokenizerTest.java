package RN.transformer;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.*;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TokenizerTest {

    private Tokenizer tokenizer;
    private WordVectors preTrainedWordVectors;
    private TransformerModel model; // Assurez-vous que cette classe existe dans votre projet
    private DataGenerator mockDataGenerator; // Assurez-vous que cette classe existe dans votre projet
    private List<DataSet> trainingData;
    private int maxSequenceLength = 9;

    @BeforeAll
    public void setUp() throws IOException {

        // Charger et former les WordVectors
        preTrainedWordVectors = trainAndSaveWordVectors("src/test/resources/word2vec/test_word2vec_model.txt");

        if (preTrainedWordVectors == null) {
            fail("Les WordVectors n'ont pas pu être chargés. Vérifiez le chemin du modèle.");
        }

        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        tokenizer = new Tokenizer(preTrainedWordVectors, embeddingSize, maxSequenceLength);

        // Ajouter des mots manquants avant d'initialiser les embeddings
        List<String> missingWords = Arrays.asList("le", "jardin", "les", "chats", "aiment", "tapis", "sur", "sol");
        for (String word : missingWords) {
            tokenizer.addToken(word);
        }

        // Réinitialiser les embeddings après avoir ajouté tous les tokens
        tokenizer.initializeEmbeddings(preTrainedWordVectors);

        // Imprimer le vocabulaire pour vérification
        tokenizer.printVocabulary();

        // Initialisation du modèle Transformer avec dModel = embeddingSize
        int numLayers = 1;
        int dModel = embeddingSize;
        int numHeads = 1;
        int dff = 512;
        int vocabSize = tokenizer.getVocabSize();
        float dropoutRate = 0.0f;
        float initialLr = 0.001f;
        int warmupSteps = 10;

        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, initialLr, warmupSteps);

        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples
        List<String> data = Arrays.asList(
            "chat mange la souris", 
            "chien court dans le jardin", 
            "les chats aiment les chiens",
            "tapis sur le sol",
            "ce film est fantastique",
            "je déteste ce temps",
            "quelle belle journée",
            "le chat sol",
            "c'est un mauvais film",
            "ce livre est intéressant",
            "je n'aime pas ce repas",
            "ce film est excellent",
            "je suis triste aujourd'hui",
            "ce temps est agréable"
        );
        List<String> targets = Arrays.asList(
            "le chat mange", 
            "le chien court", 
            "les chats aiment", 
            "le tapis sur",
            "le film est fantastique",
            "je déteste",
            "quelle belle", 
            "le chat",
            "c'est un mauvais", 
            "le livre est intéressant",
            "je n'aime pas",
            "le film est excellent",
            "je suis triste",
            "le temps est agréable"
        );
        int batchSize = 1; // Ajustez selon vos besoins
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
        
    }

    /**
     * Former et sauvegarder un modèle Word2Vec pour les tests.
     */
    private WordVectors trainAndSaveWordVectors(String modelPath) throws IOException {
        List<String> sentences = Arrays.asList(
            "<START> chat mange la souris <END>",
            "<START> chien court dans le jardin <END>",
            "<START> les chats aiment les chiens <END>",
            "<START> tapis sur le sol <END>",
            "<START> ce film est fantastique <END>",
            "<START> je déteste ce temps <END>",
            "<START> quelle belle journée <END>",
            "<START> je suis très heureux <END>",
            "<START> c'est un mauvais film <END>",
            "<START> ce livre est intéressant <END>",
            "<START> je n'aime pas ce repas <END>",
            "<START> ce film est excellent <END>",
            "<START> je suis triste aujourd'hui <END>",
            "<START> ce temps est agréable <END>"
        );

        CollectionSentenceIterator iter = new CollectionSentenceIterator(sentences);
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(10) // Augmenter les itérations pour une meilleure convergence
                .layerSize(50) // Taille des vecteurs d'embeddings
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .build();
        vec.fit();

        // Sauvegarder le modèle
        File file = new File(modelPath);
        WordVectorSerializer.writeWord2VecModel(vec, file);

        return vec;
    }

    /**
     * Préparer un petit jeu de données d'entraînement pour la classification binaire.
     * Labels : 1 pour positif, 0 pour négatif.
     */
    private void prepareTrainingData() {
        // Définir des phrases avec des labels
        // Labels : 1 pour positif, 0 pour négatif
        List<String> sentences = Arrays.asList(
            "ce film est fantastique",
            "je déteste ce temps",
            "quelle belle journée",
            "je suis très heureux",
            "c'est un mauvais film",
            "ce livre est intéressant",
            "je n'aime pas ce repas",
            "ce film est excellent",
            "je suis triste aujourd'hui",
            "ce temps est agréable"
        );
        List<Integer> labels = Arrays.asList(1, 0, 1, 1, 0, 1, 0, 1, 0, 1); // Correspondant aux phrases

        trainingData = new ArrayList<>();

        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i);
            int label = labels.get(i);

            // Tokenization
            List<String> tokens = tokenizer.tokenize(sentence);
            List<Integer> tokenIds = tokenizer.tokensToIds(tokens);
            INDArray input = tokenizer.getPretrainedEmbeddings().getRows(tokenIds.stream().mapToInt(Integer::intValue).toArray()).mean(0); // Moyenne des embeddings

            // Créer l'INDArray des features (vecteur moyen)
            // Shape : [1, embeddingSize]
            INDArray features = input;

            // Créer l'INDArray des labels (0 ou 1)
            INDArray labelArray = Nd4j.create(new double[]{label}, new int[]{1, 1});

            // Créer le DataSet
            DataSet ds = new DataSet(features, labelArray);
            trainingData.add(ds);
        }
    }

    /**
     * Construire et initialiser le modèle Transformer.
     * Vous pouvez adapter cette méthode selon la définition exacte de votre classe TransformerModel.
     */
    private void buildModel() {
        // Ici, nous supposons que TransformerModel encapsule un réseau de neurones DL4J.
        // Si ce n'est pas le cas, adaptez en conséquence.
        // Par exemple, si TransformerModel est une classe personnalisée, assurez-vous qu'elle est correctement initialisée.
        // Pour cet exemple, nous continuons d'utiliser TransformerModel tel quel.
    }


    @Test
    public void testLayerNormBackward() {
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

    /**
     * Test d'entraînement simple : vérifier que la perte diminue et l'exactitude augmente.
     */
    @Test
    public void testTrainingAndInference() throws Exception {
        

        // Entraîner le modèle sur un seul epoch
        model.train(mockDataGenerator, 2);
        //System.out.println("Initial Loss: " + initialLoss);

        // Vérification que le modèle est marqué comme entraîné
        assertTrue("Le modèle devrait être marqué comme entraîné après l'entraînement", model.isTrained());

        // Effectuer une inférence sur l'entrée d'entraînement
        String input = "le chat mange la souris";
        String actualOutput = model.infer(input, 3);
        String expectedOutput = "le chat mange";

        // Vérification que l'inférence n'est pas nulle et est cohérente
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());
        System.out.println(actualOutput);


        // Afficher les relations entre les tokens après l'entraînement
        // List<String> inputTokens = tokenizer.tokenize(input); // Ex: ["hello", "world", "input"]
        // model.displayAttentionRelations(inputTokens);

        // (Optionnel) Vérifier que l'inférence est proche de la cible
        assertEquals( expectedOutput, actualOutput,"L'inférence devrait correspondre à la cible");
    }

    @Test
    public void testEmbeddingsUpdate() throws IOException {
        
        // Récupérer l'ID du token 'chat'
        Integer chatId = tokenizer.getTokenToId().get("chat");
        assertNotNull("Le token 'chat' devrait exister dans le vocabulaire", chatId);
    
        // Copier l'embedding avant l'entraînement
        INDArray embeddingBefore = model.tokenizer.getPretrainedEmbeddings().getRow(chatId).dup();
        //System.out.println("Embedding Before:\n" + embeddingBefore);
    
        // Entraîner le modèle
        model.train(mockDataGenerator, 3);
    
        // Copier l'embedding après l'entraînement
        INDArray embeddingAfter = model.tokenizer.getPretrainedEmbeddings().getRow(chatId);
        //System.out.println("Embedding After:\n" + embeddingAfter);
    
        // Vérifier que l'embedding a changé
        boolean embeddingsChanged = !embeddingBefore.equalsWithEps(embeddingAfter, 1e-6);
        //System.out.println("Embeddings Changed: " + embeddingsChanged);
    
        // Afficher les relations entre les tokens après l'entraînement
        // List<String> inputTokens = tokenizer.tokenize("chat mange la souris"); // Inclut <START>, <END>, etc.
        // System.out.println("Input Tokens: " + inputTokens); // Ajoutez ceci pour vérifier
        // model.displayAttentionRelations(inputTokens);
    
        assertTrue("L'embedding de 'chat' devrait avoir été mis à jour.", embeddingsChanged);
    }

    /**
     * Test pour vérifier que la perte diminue au fil des epochs.
     */
    @Test
    public void testLossDecreaseOverEpochs() throws Exception {
        
        // Création d'un DataGenerator avec plusieurs batches pour simuler plusieurs epochs
        List<String> data = Arrays.asList(
            "chat manger la souris", 
            "chiens aiment le jardin", 
            "chat aiment les chiens",
            "chat sur le tapis",
            "ce livre est intéressant",
            "je n'aime pas ce repas",
            "ce film est excellent",
            "je suis triste aujourd'hui",
            "ce temps est agréable",
            "quelle belle journée"
        );
        List<String> targets = Arrays.asList(
            "le chat manger", 
            "les chiens aiment", 
            "le chat aiment", 
            "le chat sur",
            "le livre est intéressant",
            "je n'aime pas",
            "le film est excellent",
            "je suis triste",
            "le temps est agréable",
            "quelle belle"
        );
        int batchSize = 2;
        int sequenceLength = 7;
        mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, sequenceLength);

        // Initialisation des variables pour suivre la perte
        List<Float> lossHistory = new ArrayList<>();

        // Entraîner sur 10 epochs et enregistrer la perte à chaque epoch
        int numEpochs = 10;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            float loss = model.trainEpoch(mockDataGenerator);
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

    /**
     * Test d'inférence simple : prédire le label d'une nouvelle phrase.
     */
    @Test
    public void testInferenceWithPretrainedEmbeddings() throws Exception {

        // Effectuer l'entraînement
        float loss = model.trainEpoch(mockDataGenerator);
        System.out.println("Loss after training: " + loss);

        // Effectuer une inférence
        String input = "chiens aiment le jardin";
        String actualOutput = model.infer(input, 3);
        String expectedOutput = "les chiens aiment";

        // Vérifier que l'inférence est proche de la cible
        assertNotNull("L'inférence ne devrait pas être null", actualOutput);
        assertFalse("L'inférence ne devrait pas être vide", actualOutput.isEmpty());
        System.out.println("actualOutput : " + actualOutput);

        // (Optionnel) Comparer avec une sortie attendue si possible
        assertEquals(expectedOutput, actualOutput, "L'inférence devrait correspondre à la cible");
    }

    /**
     * Test pour vérifier que le vocabulaire inclut tous les mots nécessaires.
     */
    @Test
    public void testVocabularyIncludesAllWords() {
        String[] words = {"<START>", "chat", "mange", "la", "souris", "<END>", "<PAD>", "le", "chien", "court", "jardin", "les", "chats", "aiment", "tapis", "sur", "sol", "film", "temps", "belle", "journée", "heureux", "mauvais", "livre", "intéressant", "repas", "triste", "excellent", "agréable"};
        for (String word : words) {
            System.out.println("Token : " + word);
            int id = tokenizer.getTokenToId().getOrDefault(word, tokenizer.getUnkTokenId());
            assertNotEquals("Le mot '" + word + "' est mappé à <UNK>", tokenizer.getUnkTokenId(), id);
        }
    }

    /**
     * Test pour vérifier que les paramètres du modèle sont mis à jour après l'entraînement.
     */
    @Test
    public void testParameterUpdatesWithPretrainedEmbeddings() throws Exception {
        // Copier les paramètres avant l'entraînement
        List<INDArray> initialParameters = model.getCombinedParameters().stream()
                .map(INDArray::dup)
                .collect(Collectors.toList());

        // Effectuer une étape d'entraînement
        float loss = model.trainEpoch(mockDataGenerator);
        mockDataGenerator.reset();

        // Récupérer les paramètres après l'entraînement
        List<INDArray> updatedParameters = model.getCombinedParameters();

        // Vérifier que les paramètres ont été mis à jour
        for (int i = 0; i < initialParameters.size(); i++) {
            INDArray initial = initialParameters.get(i);
            INDArray updated = updatedParameters.get(i);
            assertFalse(initial.equalsWithEps(updated, 1e-6), "Les paramètres devraient avoir été mis à jour");
        }
    }
}
