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
import org.nd4j.linalg.indexing.NDArrayIndex;

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

        // Créer le vocabulaire avec uniquement les mots du fichier word2vec
        List<String> vocabulary = Arrays.asList(
            "<PAD>", "<UNK>", "<START>", "<END>",
            "le", "jardin", "dans", "manger", "aiment",
            "chiens", "les", "tapis", "sur", "est", "chat"
        );

        int embeddingSize = preTrainedWordVectors.getWordVector("chat").length;
        tokenizer = new Tokenizer(vocabulary, embeddingSize, maxSequenceLength);

        // Initialiser les embeddings avec word2vec
        tokenizer.initializeEmbeddings(preTrainedWordVectors);

        // Imprimer le vocabulaire pour vérification
        tokenizer.printVocabulary();

        // Initialisation du modèle Transformer avec dModel = embeddingSize
        int numLayers = 4;
        int dModel = embeddingSize;
        int numHeads = 2;
        int dff = 512;
        int vocabSize = tokenizer.getVocabSize();
        float dropoutRate = 0.0f;
        float initialLr = 0.001f;
        int warmupSteps = 10;

        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, initialLr, warmupSteps);

        // // Créer des listes de même taille pour les entrées et les cibles
        // List<String> data = Arrays.asList(
        //     "chat mange la", "chat dort sur", "chat joue avec",
        //     "chien court dans", "chien dort sur", "chien joue avec",
        //     "oiseau vole vers", "oiseau mange la", "oiseau dort sur"
        // );
        
        // List<String> targets = Arrays.asList(
        //     "souris", "tapis", "balle",
        //     "jardin", "tapis", "balle",
        //     "ciel", "graine", "branche"
        // );

        // int batchSize = 1;
        // mockDataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, maxSequenceLength);
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
                .iterations(5000) // Augmenter les itérations pour une meilleure convergence
                .layerSize(64) // Taille des vecteurs d'embeddings
                .seed(42)
                .windowSize(9)
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
    public void testTrainingAndInference() {
        // Configuration ultra-minimaliste
        int maxSequenceLength = 4;  // Séquences très courtes
        int dModel = 16;           // Dimension très réduite
        int numLayers = 1;         // Un seul layer
        int numHeads = 2;          // Deux têtes (diviseur de 16)
        int dff = 16;             // Même taille que dModel
        float dropoutRate = 0.0f;  // Pas de dropout
        float initialLr = 0.001f;  // Learning rate standard
        int warmupSteps = 0;       // Pas de warmup
        int epochs = 100;         // Beaucoup plus d'époques
        int batchSize = 1;         // Un exemple à la fois
        
        // Vocabulaire minimal
        List<String> vocabulary = Arrays.asList(
            "<PAD>", "<UNK>", "<START>", "<END>",
            "le", "chat", "souris"
        );
        
        // Données d'entraînement ultra-simples
        List<String> inputs = Arrays.asList(
            "le chat",
            "le chat",
            "le chat"
        );
        
        List<String> targets = Arrays.asList(
            "souris",
            "souris",
            "souris"
        );
        
        // Initialisation
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        TransformerModel model = new TransformerModel(
            numLayers, dModel, numHeads, dff, dropoutRate,
            vocabulary.size(), tokenizer, initialLr, warmupSteps
        );
        
        DataGenerator dataGenerator = new DataGenerator(
            inputs, targets, tokenizer, batchSize, maxSequenceLength
        );
        
        // Entraînement intensif
        model.train(dataGenerator, epochs);
        
        // Test
        String prediction = model.predict("le chat");
        assertEquals(prediction, "souris", "La prédiction devrait être 'souris'");
    }

    // Ajouter une méthode pour visualiser les relations grammaticales
    private void analyzeGrammaticalRelations(String input, INDArray attentionScores) {
        if (attentionScores == null) {
            System.out.println("Pas de scores d'attention disponibles");
            return;
        }

        String[] tokens = input.split(" ");
        
        // Vérifier les dimensions
        long[] shape = attentionScores.shape();
        if (shape.length < 4) {
            System.out.println("Format de scores d'attention invalide");
            return;
        }
        
        int seqLength = tokens.length;
        // Utiliser les dimensions minimales entre les tokens et les scores
        int maxI = Math.min(seqLength, (int)shape[2]);
        int maxJ = Math.min(seqLength, (int)shape[3]);

        for (int i = 0; i < maxI; i++) {
            for (int j = 0; j < maxJ; j++) {
                // Moyenne des scores sur toutes les têtes d'attention
                double score = attentionScores.get(NDArrayIndex.point(0), 
                                                 NDArrayIndex.all(), 
                                                 NDArrayIndex.point(i), 
                                                 NDArrayIndex.point(j))
                                            .meanNumber().doubleValue();
                
                if (score > 0.2) {
                    System.out.printf("Relation forte entre '%s' et '%s': %.3f%n", 
                        tokens[i], tokens[j], score);
                }
            }
        }
    }

    @Test
    public void testEmbeddingsUpdate() throws IOException {
        
        // Récupérer l'ID du token 'chat'
        Integer chatId = tokenizer.getTokenToId().get("chat");
        assertNotNull("Le token 'chat' devrait exister dans le vocabulaire", chatId);
    
        // Copier l'embedding avant l'entraînement
        INDArray embeddingBefore = model.tokenizer.getPretrainedEmbeddings().getRow(chatId).dup();
        //System.out.println("Embedding Before:\n" + embeddingBefore);
    
        // Initialiser mockDataGenerator avec des données de test
        List<String> inputs = Arrays.asList(
            "chat mange la", 
            "chat dort sur", 
            "chat joue avec"
        );
        
        List<String> targets = Arrays.asList(
            "souris",
            "tapis",
            "balle"
        );
        
        int batchSize = 1;
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, maxSequenceLength);

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

        // Vrification que la perte diminue au fil des epochs
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
        // Créer un nouveau tokenizer avec le vocabulaire exact du fichier word2vec
        List<String> vocabulary = Arrays.asList(
            "<PAD>", "<UNK>", "<START>", "<END>",
            "le", "jardin", "dans", "manger", "aiment",
            "chiens", "les", "tapis", "sur", "est", "chat"
        );
        
        int dModel = preTrainedWordVectors.getWordVector("chat").length;
        Tokenizer newTokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);
        newTokenizer.initializeEmbeddings(preTrainedWordVectors);
        
        // Vérifier que tous les tokens sont dans le vocabulaire
        for (String token : vocabulary) {
            int id = newTokenizer.getTokenToId().get(token);
            System.out.println("Token '" + token + "' -> ID: " + id);
        }
        
        // Données d'entraînement avec des mots du vocabulaire
        List<String> inputs = Arrays.asList(
            "chat aiment", "chat aiment", "chat aiment", "chat aiment"
        );
        
        List<String> targets = Arrays.asList(
            "chiens", "chiens", "chiens", "chiens"
        );
        
        // Créer un nouveau modèle avec le nouveau tokenizer
        model = new TransformerModel(2, dModel, 2, 128, 0.1f, newTokenizer.getVocabSize(), 
                                    newTokenizer, 0.01f, 2);
        
        int batchSize = 2;
        DataGenerator dataGen = new DataGenerator(inputs, targets, newTokenizer, batchSize, maxSequenceLength);
        
        // Entraîner le modèle
        model.train(dataGen, 20);
        
        // Test
        String input = "chat aiment";
        String actualOutput = model.infer(input, 1);
        String expectedOutput = "chiens";

        assertEquals(expectedOutput, actualOutput.trim(), 
                    "L'inférence devrait correspondre à la cible");
    }

    /**
     * Test pour vérifier que le vocabulaire inclut tous les mots nécessaires.
     */
    @Test
    public void testVocabularyIncludesAllWords() {
        // Utiliser uniquement les mots qui sont dans le fichier word2vec
        String[] words = {
            "<START>", "<END>", "<PAD>",
            "le", "jardin", "dans", "manger", "aiment",
            "chiens", "les", "tapis", "sur", "est", "chat"
        };
        
        for (String word : words) {
            System.out.println("Token : " + word);
            int id = tokenizer.getTokenToId().getOrDefault(word, tokenizer.getUnkTokenId());
            assertNotEquals("Le mot '" + word + "' devrait être dans le vocabulaire", 
                           tokenizer.getUnkTokenId(), id);
        }
    }

    /**
     * Test pour vérifier que les paramètres du modèle sont mis à jour après l'entraînement.
     */
    @Test
    public void testParameterUpdatesWithPretrainedEmbeddings() throws Exception {
        // Initialiser mockDataGenerator avant l'entraînement
        List<String> inputs = Arrays.asList(
            "chat mange la", 
            "chat dort sur"
        );
        
        List<String> targets = Arrays.asList(
            "souris",
            "tapis"
        );
        
        int batchSize = 1;
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, maxSequenceLength);

        // Copier les paramètres avant l'entraînement
        List<INDArray> initialParameters = model.getCombinedParameters().stream()
                .map(INDArray::dup)
                .collect(Collectors.toList());

        // Effectuer une étape d'entraînement
        float loss = model.trainEpoch(mockDataGenerator);
        
        // Récupérer les paramètres après l'entraînement
        List<INDArray> updatedParameters = model.getCombinedParameters();

        // Vérifier que les paramètres ont été mis à jour
        for (int i = 0; i < initialParameters.size(); i++) {
            INDArray initial = initialParameters.get(i);
            INDArray updated = updatedParameters.get(i);
            assertFalse(initial.equalsWithEps(updated, 1e-6), "Les paramètres devraient avoir été mis à jour");
        }
    }

    @Test
    public void testTrainingWithPretrainedEmbeddings() throws Exception {
        // Données d'entraînement simplifiées et répétées pour un meilleur apprentissage
        List<String> inputs = Arrays.asList(
            "chat manger", "chat manger", "chat manger", "chat manger",
            "chat manger", "chat manger", "chat manger", "chat manger"
        );
        
        List<String> targets = Arrays.asList(
            "les tapis", "les tapis", "les tapis", "les tapis",
            "les tapis", "les tapis", "les tapis", "les tapis"
        );
        
        int batchSize = 2;
        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, maxSequenceLength);
        
        // Configuration du modèle avec des paramètres optimisés
        int dModel = preTrainedWordVectors.getWordVector("chat").length;
        model = new TransformerModel(
            4,          // Plus de couches
            dModel, 
            4,          // Plus de têtes d'attention
            512,        // FFN plus large
            0.1f,       // Dropout
            tokenizer.getVocabSize(), 
            tokenizer, 
            0.0005f,    // Learning rate plus faible
            10          // Plus de warmup steps
        );
        
        // Entraîner plus longtemps
        model.train(mockDataGenerator, 100);  // Plus d'époques
        
        // Test avec une tolérance pour les prédictions partielles
        String input = "chat manger";
        String actualOutput = model.infer(input, 2);
        assertTrue(
            actualOutput.contains("les") || actualOutput.contains("tapis"),
            "La prédiction devrait contenir au moins un des mots cibles"
        );
    }
}
