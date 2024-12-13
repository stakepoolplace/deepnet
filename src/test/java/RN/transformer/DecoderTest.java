package RN.transformer;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.transformer.Decoder.DecoderLayer;
import RN.utils.NDArrayUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class DecoderTest {

    private Decoder decoder;
    private Tokenizer tokenizer;
    private INDArray encodedInput;
    private INDArray encoderInputTokens;
    private DataGenerator mockDataGenerator;
    private TransformerModel model;

    private int dModel = 4; // Utilisation d'une dimension réduite pour simplifier le test
    private int maxSequenceLength = 3;
    private int numLayers = 1; // Utilisation d'une seule couche pour simplifier
    private int numHeads = 1; // Utilisation d'une seule tête
    private int dff = 4 * dModel; // standard
    private double dropoutRate = 0.0;
    private float initialLr = 0.001f;
    private int warmupSteps = 10;
    private int batchSize = 1;
    private int sequenceLength = 3;

    @Before
    public void setUp() {

        // Initialisation du Tokenizer avec un vocabulaire simple
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");

        tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);

        model = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabulary.size(), tokenizer,
                initialLr, warmupSteps, true);
        // Création d'un DataGenerator fictif avec des paires d'entrée-cible simples
        List<String> inputs = Arrays.asList(
                "hello",
                "world");
        List<String> targets = Arrays.asList(
                "world",
                "hello");

        mockDataGenerator = new DataGenerator(inputs, targets, tokenizer, batchSize, sequenceLength);

        decoder = model.getDecoder();

        // Simuler une entrée encodée avec des valeurs connues
        encodedInput = Nd4j.create(new float[][][] {
                {
                        { 1.0f, 0.0f, 0.0f, 0.0f }, // Embedding pour <START>
                        { 0.0f, 1.0f, 0.0f, 0.0f }, // Embedding pour "hello"
                        { 0.0f, 0.0f, 1.0f, 0.0f } // Embedding pour "world"
                }
        });
        encoderInputTokens = Nd4j.create(new int[][] {
                { tokenizer.getStartTokenId(), tokenizer.getPadTokenId(), tokenizer.getEndTokenId() }
        });

        // Initialiser les poids du décodeur à des valeurs déterministes
        initializeDecoderWeights(decoder);
    }

    /**
     * Initialise les poids du décodeur et de ses sous-couches à des valeurs connues
     * pour le test.
     */
    private void initializeDecoderWeights(Decoder decoder) {
        // Pour chaque couche du décodeur
        for (Decoder.DecoderLayer layer : decoder.layers) {
            // Initialiser les poids de l'attention multi-tête
            layer.selfAttention.setWq(Nd4j.eye(dModel));
            layer.selfAttention.setWk(Nd4j.eye(dModel));
            layer.selfAttention.setWv(Nd4j.eye(dModel));
            layer.selfAttention.setWo(Nd4j.eye(dModel));

            layer.encoderDecoderAttention.setWq(Nd4j.eye(dModel));
            layer.encoderDecoderAttention.setWk(Nd4j.eye(dModel));
            layer.encoderDecoderAttention.setWv(Nd4j.eye(dModel));
            layer.encoderDecoderAttention.setWo(Nd4j.eye(dModel));

            // Initialiser les poids du feed-forward pour scaler x par 8.0f
            PositionwiseFeedForward feedForward = layer.feedForward;
            INDArray W1 = Nd4j.eye(dModel).mul(2.0f); // W1 = 2 * I
            INDArray W2 = Nd4j.eye(dModel).mul(4.0f); // W2 = 4 * I
            INDArray b1 = Nd4j.zeros(1, dModel);
            INDArray b2 = Nd4j.zeros(1, dModel);
            feedForward.setW1(W1);
            feedForward.setW2(W2);
            feedForward.setB1(b1);
            feedForward.setB2(b2);

        }

        // Initialiser les poids de la projection linéaire du décodeur
        decoder.linearProjection.setWeights(NDArrayUtils.eye(dModel, tokenizer.getVocabSize())); // [dModel, vocabSize]
        decoder.linearProjection.setBias(Nd4j.zeros(1, tokenizer.getVocabSize()));

        // Initialiser les embeddings du décodeur
        tokenizer.initializeEmbeddingEye();
    }

    @Test
    public void testProject() {
        // Initialiser LinearProjection avec LayerNorm désactivé pour ce test
        LinearProjection lp = new LinearProjection(4, 6, false); // dModel=4, vocabSize=6, useLayerNorm=false

        // Initialiser les poids à une matrice identité étendue et biais à zéro
        INDArray identityWeights = NDArrayUtils.eye(4, 6); // [4,6]
        INDArray zeroBias = Nd4j.zeros(1, 6); // [1,6]
        lp.setParameters(identityWeights, zeroBias, null, null);

        // Créer une entrée de rang 3 [batchSize=3, seqLength=4, inputSize=4]
        INDArray input = Nd4j.create(new float[][][] {
                {
                        { 8.0f, 0.0f, 0.0f, 0.0f }, // Projection pour <START>
                        { 0.0f, 8.0f, 0.0f, 0.0f }, // Projection pour "hello"
                        { 0.0f, 0.0f, 8.0f, 0.0f }, // Projection pour "world"
                        { 0.0f, 0.0f, 0.0f, 8.0f } // Projection supplémentaire si nécessaire
                },
                {
                        { 8.0f, 0.0f, 0.0f, 0.0f },
                        { 0.0f, 8.0f, 0.0f, 0.0f },
                        { 0.0f, 0.0f, 8.0f, 0.0f },
                        { 0.0f, 0.0f, 0.0f, 8.0f }
                },
                {
                        { 8.0f, 0.0f, 0.0f, 0.0f },
                        { 0.0f, 8.0f, 0.0f, 0.0f },
                        { 0.0f, 0.0f, 8.0f, 0.0f },
                        { 0.0f, 0.0f, 0.0f, 8.0f }
                }
        }); // [3,4,4]

        // Effectuer la projection
        INDArray output = lp.project(input); // [3,4,6]

        // Vérifier la forme de la sortie : [3, 4, 6]
        assertArrayEquals("Output shape should be [3, 4, 6]", new long[] { 3, 4, 6 }, output.shape());

        // Vérifier le type de données
        assertEquals("Output data type should be FLOAT", DataType.FLOAT, output.dataType());

        // Afficher les poids et biais de la projection linéaire
        System.out.println("===== LinearProjection Weights =====");
        System.out.println(lp.getWeights());
        System.out.println("===== LinearProjection Bias =====");
        System.out.println(lp.getBias());

        // Afficher les valeurs d'entrée après reshaping et scaling
        System.out.println("===== Input After Reshaping and Scaling =====");
        System.out.println(input);

        // Afficher la sortie après projection linéaire
        System.out.println("===== Output After Linear Projection =====");
        System.out.println(output);

        // Vérifier les valeurs attendues
        INDArray expectedLogitsManualCorrect = Nd4j.create(new double[][][] {
                {
                        { 8.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // <START> projection
                        { 0.0, 8.0, 0.0, 0.0, 0.0, 0.0 }, // "hello" projection
                        { 0.0, 0.0, 8.0, 0.0, 0.0, 0.0 }, // "world" projection
                        { 0.0, 0.0, 0.0, 8.0, 0.0, 0.0 } // Projection supplémentaire si nécessaire
                },
                {
                        { 8.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 8.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 8.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 8.0, 0.0, 0.0 }
                },
                {
                        { 8.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 8.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 8.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 8.0, 0.0, 0.0 }
                }
        }); // [3,4,6]

        // Comparer les logits réels avec les logits attendus
        double epsilon = 1e-5;
        boolean allMatch = true;
        StringBuilder mismatchDetails = new StringBuilder();
        for (int i = 0; i < expectedLogitsManualCorrect.size(0); i++) {
            for (int j = 0; j < expectedLogitsManualCorrect.size(1); j++) {
                for (int k = 0; k < expectedLogitsManualCorrect.size(2); k++) {
                    double expected = expectedLogitsManualCorrect.getDouble(i, j, k);
                    double actual = output.getDouble(i, j, k);
                    if (Math.abs(expected - actual) > epsilon) {
                        allMatch = false;
                        mismatchDetails.append(String.format("Mismatch at (%d,%d,%d): expected %.6f but was %.6f%n", i,
                                j, k, expected, actual));
                    }
                }
            }
        }

        if (!allMatch) {
            fail("Les logits ne correspondent pas aux valeurs attendues:\n" + mismatchDetails.toString());
        }

        // Si tout correspond, afficher un message de succès
        System.out.println("Tous les logits correspondent aux valeurs attendues.");

        // Ajouter des traces supplémentaires pour les composants internes si nécessaire
        // Par exemple, si Decoder a des méthodes pour récupérer les couches, faire une
        // boucle
        // et afficher les poids des sous-couches
        System.out.println("===== Vérification des Couches Internes du Decoder =====");

        // Supposons que Decoder a une liste de DecoderLayer accessible
        for (int layerIdx = 0; layerIdx < decoder.layers.size(); layerIdx++) {
            DecoderLayer layer = decoder.layers.get(layerIdx);
            System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Self-Attention Weights =====");
            System.out.println("Wq:");
            System.out.println(layer.selfAttention.getWq());
            System.out.println("Wk:");
            System.out.println(layer.selfAttention.getWk());
            System.out.println("Wv:");
            System.out.println(layer.selfAttention.getWv());
            System.out.println("Wo:");
            System.out.println(layer.selfAttention.getWo());

            System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Encoder-Decoder Attention Weights =====");
            System.out.println("Wq:");
            System.out.println(layer.encoderDecoderAttention.getWq());
            System.out.println("Wk:");
            System.out.println(layer.encoderDecoderAttention.getWk());
            System.out.println("Wv:");
            System.out.println(layer.encoderDecoderAttention.getWv());
            System.out.println("Wo:");
            System.out.println(layer.encoderDecoderAttention.getWo());

            System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Feed-Forward Weights =====");
            System.out.println("W1:");
            System.out.println(layer.feedForward.getW1());
            System.out.println("W2:");
            System.out.println(layer.feedForward.getW2());
            System.out.println("b1:");
            System.out.println(layer.feedForward.getB1());
            System.out.println("b2:");
            System.out.println(layer.feedForward.getB2());
        }

        // Afficher les poids et biais de la projection linéaire du décodeur
        System.out.println("===== Decoder LinearProjection Weights =====");
        System.out.println(decoder.linearProjection.getWeights());
        System.out.println("===== Decoder LinearProjection Bias =====");
        System.out.println(decoder.linearProjection.getBias());
    }

    @Test
    public void testLinearProjectionWithKnownWeights() {
        // Initialiser LinearProjection avec LayerNorm désactivé
        LinearProjection lp = new LinearProjection(3, 2, false); // inputSize=3, outputSize=2, useLayerNorm=false

        // Initialiser les poids à une matrice simple et biais à zéro
        INDArray weights = Nd4j.create(new float[][] {
                { 1.0f, 0.0f },
                { 0.0f, 1.0f },
                { 1.0f, 1.0f }
        }); // [3,2]
        INDArray bias = Nd4j.zeros(1, 2); // [1,2]
        lp.setParameters(weights, bias, null, null);

        // Créer une entrée de rang 3 [batchSize=2, seqLength=2, inputSize=3]
        INDArray input = Nd4j.create(new float[][][] {
                {
                        { 1.0f, 2.0f, 3.0f },
                        { 4.0f, 5.0f, 6.0f }
                },
                {
                        { 7.0f, 8.0f, 9.0f },
                        { 10.0f, 11.0f, 12.0f }
                }
        }); // [2,2,3]

        // Effectuer la projection
        INDArray output = lp.project(input); // [2,2,2]

        // Définir la sortie attendue
        INDArray expected = Nd4j.create(new float[][][] {
                {
                        { 1.0f * 1f + 2.0f * 0f + 3.0f * 1f, 1.0f * 0f + 2.0f * 1f + 3.0f * 1f }, // {4.0, 5.0}
                        { 4.0f * 1f + 5.0f * 0f + 6.0f * 1f, 4.0f * 0f + 5.0f * 1f + 6.0f * 1f } // {10.0, 11.0}
                },
                {
                        { 7.0f * 1f + 8.0f * 0f + 9.0f * 1f, 7.0f * 0f + 8.0f * 1f + 9.0f * 1f }, // {16.0, 17.0}
                        { 10.0f * 1f + 11.0f * 0f + 12.0f * 1f, 10.0f * 0f + 11.0f * 1f + 12.0f * 1f } // {22.0, 23.0}
                }
        }); // [2,2,2]

        // Comparer les valeurs réelles avec les valeurs attendues
        double epsilon = 1e-5;
        boolean allMatch = true;
        StringBuilder mismatchDetails = new StringBuilder();
        for (int i = 0; i < expected.size(0); i++) {
            for (int j = 0; j < expected.size(1); j++) {
                for (int k = 0; k < expected.size(2); k++) {
                    double expectedVal = expected.getDouble(i, j, k);
                    double actualVal = output.getDouble(i, j, k);
                    if (Math.abs(expectedVal - actualVal) > epsilon) {
                        allMatch = false;
                        mismatchDetails.append(String.format("Mismatch at (%d,%d,%d): expected %.6f but was %.6f%n", i,
                                j, k, expectedVal, actualVal));
                    }
                }
            }
        }

        if (!allMatch) {
            fail("Les logits de la projection linéaire ne correspondent pas aux valeurs attendues:\n"
                    + mismatchDetails.toString());
        }

        // Si tout correspond, afficher un message de succès
        System.out.println("La projection linéaire fonctionne correctement avec les poids connus.");
    }

    @Test
    public void testDecoderFunctionality() {
        // Initialiser les IDs de sortie avec le token de début, pad, end
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getPadTokenId(), tokenizer.getEndTokenId());
    
        // Convertir les IDs de sortie en INDArray 2D
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });
    
        // Créer un Batch pour le décodeur
        Batch decoderBatch = new Batch(decoderInputIds, null, tokenizer);
    
        // Créer les masques nécessaires
        INDArray queryPaddingMaskFromSource = NDArrayUtils.createQueryPaddingMask(tokenizer, encoderInputTokens);
        INDArray keyPaddingMaskFromSource = NDArrayUtils.createKeyPaddingMask(tokenizer, encoderInputTokens);
        INDArray queryPaddingMaskFromTarget = NDArrayUtils.createQueryPaddingMask(tokenizer, decoderInputIds);
        INDArray keyPaddingMaskFromTarget = NDArrayUtils.createKeyPaddingMask(tokenizer, decoderInputIds);
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask((int) decoderInputIds.shape()[0],
                (int) decoderInputIds.shape()[1]);
    
    
            System.out.println("===== Vérification des Couches Internes du Decoder =====");
    
            // Supposons que Decoder a une liste de DecoderLayer accessible
            for (int layerIdx = 0; layerIdx < decoder.layers.size(); layerIdx++) {
                DecoderLayer layer = decoder.layers.get(layerIdx);
                System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Self-Attention Weights =====");
                System.out.println("Wq:");
                System.out.println(layer.selfAttention.getWq());
                System.out.println("Wk:");
                System.out.println(layer.selfAttention.getWk());
                System.out.println("Wv:");
                System.out.println(layer.selfAttention.getWv());
                System.out.println("Wo:");
                System.out.println(layer.selfAttention.getWo());
    
                System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Encoder-Decoder Attention Weights =====");
                System.out.println("Wq:");
                System.out.println(layer.encoderDecoderAttention.getWq());
                System.out.println("Wk:");
                System.out.println(layer.encoderDecoderAttention.getWk());
                System.out.println("Wv:");
                System.out.println(layer.encoderDecoderAttention.getWv());
                System.out.println("Wo:");
                System.out.println(layer.encoderDecoderAttention.getWo());
    
                System.out.println("===== Decoder Layer " + (layerIdx + 1) + " Feed-Forward Weights =====");
                System.out.println("W1:");
                System.out.println(layer.feedForward.getW1());
                System.out.println("W2:");
                System.out.println(layer.feedForward.getW2());
                System.out.println("b1:");
                System.out.println(layer.feedForward.getB1());
                System.out.println("b2:");
                System.out.println(layer.feedForward.getB2());
            }
    
    
                
    
        // Décoder en passant les tokens d'entrée de l'encodeur
        System.out.println("===== Starting Decode =====");
        INDArray logits = decoder.decode(false,
                encodedInput,
                encodedInput,
                decoderBatch,
                lookAheadMask,
                queryPaddingMaskFromSource,
                keyPaddingMaskFromSource,
                queryPaddingMaskFromTarget,
                keyPaddingMaskFromTarget);
        System.out.println("===== Decode Completed =====");
        System.out.println("Logits: ");
        System.out.println(logits);
    
        // Vérifier que les logits ne sont pas nuls
        assertNotNull("Les logits ne devraient pas être null", logits);
    
        // Vérifier la forme des logits
        assertArrayEquals("La forme des logits devrait être [batchSize, seqLength, vocabSize]",
                new long[] { 1, 3, 6 }, logits.shape());
    


        // Définir les logits attendus après LayerNorm et projection linéaire
        // Basé sur les calculs manuels ci-dessus
        INDArray expectedLogitsManualCorrect = Nd4j.create(new double[][][] {
            {
                { -0.5697, -0.5697, 1.7320, -0.5925, 0.0, 0.0 }, // <START>
                { 1.7321, -0.5774, -0.5774, -0.5774, 0.0, 0.0 }, // <PAD>
                { -0.5446, -0.6414, -0.5446, 1.7307, 0.0, 0.0 }  // <END>
            }
        }); // [1,3,6]
    
        // Comparer les logits réels avec les logits attendus
        double epsilon = 1e-4; // précision adapté aux imprécisions de calcul manuel
        boolean allMatch = true;
        StringBuilder mismatchDetails = new StringBuilder();
        for (int i = 0; i < expectedLogitsManualCorrect.size(0); i++) {
            for (int j = 0; j < expectedLogitsManualCorrect.size(1); j++) {
                for (int k = 0; k < expectedLogitsManualCorrect.size(2); k++) {
                    double expectedVal = expectedLogitsManualCorrect.getDouble(i, j, k);
                    double actualVal = logits.getDouble(i, j, k);
                    if (Math.abs(expectedVal - actualVal) > epsilon) {
                        allMatch = false;
                        mismatchDetails.append(String.format("Mismatch at (%d,%d,%d): expected %.6f but was %.6f%n",
                                i, j, k, expectedVal, actualVal));
                    }
                }
            }
        }
    
        if (!allMatch) {
            fail("Les logits de la projection linéaire ne correspondent pas aux valeurs attendues:\n" + mismatchDetails.toString());
        }
    
        // Si tout correspond, afficher un message de succès
        System.out.println("Les logits correspondent aux attentes avec une tolérance ajustée.");
    }
    
    @Test
    public void testDecoderFunctionalityWithLayerNorm() {
        // Initialiser les IDs de sortie avec le token de début, pad, end
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getPadTokenId(), tokenizer.getEndTokenId());
    
        // Convertir les IDs de sortie en INDArray 2D
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });
    
        // Créer un Batch pour le décodeur
        Batch decoderBatch = new Batch(decoderInputIds, null, tokenizer);
    
        // Créer les masques nécessaires
        INDArray queryPaddingMaskFromSource = NDArrayUtils.createQueryPaddingMask(tokenizer, encoderInputTokens);
        INDArray keyPaddingMaskFromSource = NDArrayUtils.createKeyPaddingMask(tokenizer, encoderInputTokens);
        INDArray queryPaddingMaskFromTarget = NDArrayUtils.createQueryPaddingMask(tokenizer, decoderInputIds);
        INDArray keyPaddingMaskFromTarget = NDArrayUtils.createKeyPaddingMask(tokenizer, decoderInputIds);
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask((int) decoderInputIds.shape()[0],
                (int) decoderInputIds.shape()[1]);
    
        // Décoder en passant les tokens d'entrée de l'encodeur
        System.out.println("===== Starting Decode =====");
        INDArray logits = decoder.decode(false,
                encodedInput,
                encodedInput,
                decoderBatch,
                lookAheadMask,
                queryPaddingMaskFromSource,
                keyPaddingMaskFromSource,
                queryPaddingMaskFromTarget,
                keyPaddingMaskFromTarget);
        System.out.println("===== Decode Completed =====");
        System.out.println("Logits: ");
        System.out.println(logits);
    
        // Vérifier que les logits ne sont pas nuls
        assertNotNull("Les logits ne devraient pas être null", logits);
    
        // Vérifier la forme des logits
        assertEquals("La forme des logits devrait être [batchSize, seqLength, vocabSize]",
                3, logits.shape().length);
    
        // Vérifier les propriétés de LayerNorm sur les premières 4 dimensions
        for (int i = 0; i < logits.size(0); i++) {
            for (int j = 0; j < logits.size(1); j++) {
                // Accéder correctement au vecteur de logits [vocabSize]
                INDArray logitVector = logits.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all());
    
                // Séparer les premières 4 dimensions et les deux dernières
                INDArray first4 = logitVector.get(NDArrayIndex.interval(0, 4));
                INDArray last2 = logitVector.get(NDArrayIndex.interval(4, 6));
    
                // Calculer la moyenne et la variance des premières 4 dimensions
                double meanFirst4 = first4.meanNumber().doubleValue();
                double varianceFirst4 = first4.var(false).getDouble(0); // Utiliser varNumber(false)
    
                // Afficher les moyennes et variances pour diagnostic
                System.out.printf("Logit Vector (%d, %d): Mean (first4) = %.4f, Variance (first4) = %.4f%n", 
                                  i, j, meanFirst4, varianceFirst4);
    
                // Vérifier que la moyenne des premières 4 dimensions est proche de 0
                assertEquals("La moyenne des premières 4 dimensions des logits devrait être proche de 0", 
                             0.0, meanFirst4, 1e-1);
    
                // Vérifier que la variance des premières 4 dimensions des logits est proche de 1.3333
                assertEquals("La variance des premières 4 dimensions des logits devrait être proche de 1", 
                             1.0, varianceFirst4, 1e-4);
    
                // Vérifier que les deux dernières dimensions sont bien des zéros
                double last2Val0 = last2.getDouble(0);
                double last2Val1 = last2.getDouble(1);
    
                assertEquals("La cinquième dimension des logits devrait être 0", 0.0, last2Val0, 1e-5);
                assertEquals("La sixième dimension des logits devrait être 0", 0.0, last2Val1, 1e-5);
            }
        }
    
        // Si tout correspond, afficher un message de succès
        System.out.println("La projection linéaire avec LayerNorm fonctionne correctement.");
    }

    @Test
    public void testAttentionMechanism() {
        // Initialiser les IDs de sortie avec une séquence valide
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getUnkTokenId(),
                tokenizer.getEndTokenId());
    
        // Convertir les IDs de sortie en INDArray 2D avec la bonne forme [1, seqLength]
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });
    
        // Créer un Batch pour le décodeur avec les tokens d'entrée
        Batch decoderBatch = new Batch(decoderInputIds, decoderInputIds, tokenizer);
    
        // Créer les masques nécessaires (ici, supposons qu'ils sont nulls pour simplifier)
        INDArray queryPaddingMaskFromSource = NDArrayUtils.createQueryPaddingMask(tokenizer, encoderInputTokens);
        INDArray keyPaddingMaskFromSource = NDArrayUtils.createKeyPaddingMask(tokenizer, encoderInputTokens);
        INDArray queryPaddingMaskFromTarget = NDArrayUtils.createQueryPaddingMask(tokenizer, decoderInputIds);
        INDArray keyPaddingMaskFromTarget = NDArrayUtils.createKeyPaddingMask(tokenizer, decoderInputIds);
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask((int) decoderInputIds.shape()[0],
                (int) decoderInputIds.shape()[1]);
    
        // Initialiser les poids à des matrices d'identité pour un calcul déterministe
        MultiHeadAttention multiHeadAttention = new MultiHeadAttention(4, 1); // dModel=4, numHeads=1
        INDArray Wq = NDArrayUtils.eye(4, 4);
        INDArray Wk = NDArrayUtils.eye(4, 4);
        INDArray Wv = NDArrayUtils.eye(4, 4);
        INDArray Wo = NDArrayUtils.eye(4, 4);
        multiHeadAttention.setWeights(Wq, Wk, Wv, Wo);
    
        // Définir l'entrée
        INDArray input = Nd4j.create(new float[][][] {
            {
                {1.0f, 2.0f, 3.0f, 4.0f},
                {2.0f, 3.0f, 4.0f, 5.0f}
            }
        }); // [1,2,4]
    
        // Appliquer LayerNorm
        LayerNorm layerNorm = new LayerNorm(4);
        INDArray normalizedInput = layerNorm.forward(input); // Moyenne=2.5, variance=1.25
    
        // Appliquer MultiHeadAttention
        INDArray attentionOutput = multiHeadAttention.forwardSelfAttention(normalizedInput, null, null); // [1,2,4]
    
        // Appliquer la projection linéaire pour obtenir les logits [1,2,6]
        // Initialiser une projection linéaire déterministe (identité étendue)
        LinearProjection linearProjection = new LinearProjection(4, 6, false); // dModel=4, vocabSize=6, useLayerNorm=false
        INDArray projectionWeights = NDArrayUtils.eye(4, 6); // [4,6] identité étendue
        INDArray projectionBias = Nd4j.zeros(1, 6); // [1,6] biais à zéro
        linearProjection.setParameters(projectionWeights, projectionBias, null, null);
    
        // Effectuer la projection
        INDArray logits = linearProjection.project(attentionOutput); // [1,2,6]
    
        // Définir les logits attendus basés sur les poids identitaires étendus
        // Avec W = [I_4 | 0_4x2], les logits devraient être les mêmes que l'attentionOutput étendues avec deux zéros
        INDArray expectedLogitsManualCorrect = Nd4j.create(new double[][][] {
            {
                { -1.342, -0.447, 0.447, 1.342, 0.0, 0.0 }, // <START>
                { -1.342, -0.447, 0.447, 1.342, 0.0, 0.0 }  // <UNK>
            }
        }); // [1,2,6]
    
        // Comparer les logits réels avec les logits attendus
        double epsilon = 1e-3; // Tolérance ajustée
        boolean allMatch = true;
        StringBuilder mismatchDetails = new StringBuilder();
        for (int i = 0; i < expectedLogitsManualCorrect.size(0); i++) {
            for (int j = 0; j < expectedLogitsManualCorrect.size(1); j++) {
                for (int k = 0; k < expectedLogitsManualCorrect.size(2); k++) {
                    double expectedVal = expectedLogitsManualCorrect.getDouble(i, j, k);
                    double actualVal = logits.getDouble(i, j, k);
                    if (Math.abs(expectedVal - actualVal) > epsilon) {
                        allMatch = false;
                        mismatchDetails.append(String.format(
                            "Mismatch at (%d,%d,%d): expected %.6f but was %.6f%n",
                            i, j, k, expectedVal, actualVal));
                    }
                }
            }
        }
    
        if (!allMatch) {
            fail("Les logits ne correspondent pas aux valeurs attendues:\n" + mismatchDetails.toString());
        }
    
        // Si tout correspond, afficher un message de succès
        System.out.println("Les poids d'attention sont normalisés et les logits correspondent aux attentes connues avec une tolérance.");
    }

    @Test
    public void testCrossAttentionMechanism() {
        // Initialiser les IDs de sortie avec une séquence valide
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getUnkTokenId(),
                tokenizer.getEndTokenId());
    
        // Convertir les IDs de sortie en INDArray 2D avec la bonne forme [1, seqLength]
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });
    
        // Créer un Batch pour le décodeur avec les tokens d'entrée
        Batch decoderBatch = new Batch(decoderInputIds, decoderInputIds, tokenizer);
    
        // Créer les masques nécessaires
        INDArray queryPaddingMaskFromSource = NDArrayUtils.createQueryPaddingMask(tokenizer, encoderInputTokens);
        INDArray keyPaddingMaskFromSource = NDArrayUtils.createKeyPaddingMask(tokenizer, encoderInputTokens);
        INDArray queryPaddingMaskFromTarget = NDArrayUtils.createQueryPaddingMask(tokenizer, decoderInputIds);
        INDArray keyPaddingMaskFromTarget = NDArrayUtils.createKeyPaddingMask(tokenizer, decoderInputIds);
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask((int) decoderInputIds.shape()[0],
                (int) decoderInputIds.shape()[1]);
    
        // Initialiser les poids à des matrices d'identité pour un calcul déterministe
        MultiHeadAttention selfHeadAttention = new MultiHeadAttention(4, 1); // dModel=4, numHeads=1
        INDArray Wq_self = NDArrayUtils.eye(4, 4);
        INDArray Wk_self = NDArrayUtils.eye(4, 4);
        INDArray Wv_self = NDArrayUtils.eye(4, 4);
        INDArray Wo_self = NDArrayUtils.eye(4, 4);
        selfHeadAttention.setWeights(Wq_self, Wk_self, Wv_self, Wo_self);
    
        MultiHeadAttention crossHeadAttention = new MultiHeadAttention(4, 1); // dModel=4, numHeads=1
        INDArray Wq_cross = NDArrayUtils.eye(4, 4);
        INDArray Wk_cross = NDArrayUtils.eye(4, 4);
        INDArray Wv_cross = NDArrayUtils.eye(4, 4);
        INDArray Wo_cross = NDArrayUtils.eye(4, 4);
        crossHeadAttention.setWeights(Wq_cross, Wk_cross, Wv_cross, Wo_cross);
    
        // Définir l'entrée pour le décodeur (requêtes)
        INDArray decoderInput = Nd4j.create(new float[][][] {
            {
                {1.0f, 2.0f, 3.0f, 4.0f},
                {2.0f, 3.0f, 4.0f, 5.0f}
            }
        }); // [1,2,4]
    
        // Définir l'entrée pour l'encodeur (clés et valeurs)
        INDArray encoderInput = Nd4j.create(new float[][][] {
            {
                {5.0f, 6.0f, 7.0f, 8.0f},
                {6.0f, 7.0f, 8.0f, 9.0f},
                {7.0f, 8.0f, 9.0f, 10.0f}
            }
        }); // [1,3,4]
    
        // Appliquer LayerNorm sur les entrées
        LayerNorm layerNormDecoder = new LayerNorm(4);
        INDArray normalizedDecoderInput = layerNormDecoder.forward(decoderInput); // Moyenne et variance calculées
    
        LayerNorm layerNormEncoder = new LayerNorm(4);
        INDArray normalizedEncoderInput = layerNormEncoder.forward(encoderInput); // Moyenne et variance calculées
    
        // Appliquer Self-Attention
        INDArray selfAttentionOutput = selfHeadAttention.forwardSelfAttention(normalizedDecoderInput, null, null); // [1,2,4]
    
        // Appliquer Cross-Attention en utilisant la sortie de l'encodeur
        INDArray crossAttentionOutput = crossHeadAttention.forwardCrossAttention(
            selfAttentionOutput,       // query
            normalizedEncoderInput,    // key
            normalizedEncoderInput,    // value
            null,                      // queryMask
            null                       // keyMask
        ); // [1,2,4]
    
        // Tracer les poids d'attention cross
        System.out.println("===== Cross-Attention Weights =====");
        INDArray crossAttentionWeights = crossHeadAttention.getAttentionWeights(); // [batchSize, numHeads, seqLength_target, seqLength_source]
        System.out.println(crossAttentionWeights);
    
        // Appliquer la projection linéaire pour obtenir les logits [1,2,6]
        // Initialiser une projection linéaire déterministe (identité étendue)
        LinearProjection linearProjection = new LinearProjection(4, 6, false); // dModel=4, vocabSize=6, useLayerNorm=false
        INDArray projectionWeights = NDArrayUtils.eye(4, 6); // [4,6] identité étendue
        INDArray projectionBias = Nd4j.zeros(1, 6); // [1,6] biais à zéro
        linearProjection.setParameters(projectionWeights, projectionBias, null, null);
    
        // Effectuer la projection
        INDArray logits = linearProjection.project(crossAttentionOutput); // [1,2,6]
    
        // Définir les logits attendus basés sur les poids identitaires étendus
        // Avec W = [I_4 | 0_4x2], les logits devraient être les mêmes que l'attentionOutput étendues avec deux zéros
        INDArray expectedLogitsManualCorrect = Nd4j.create(new double[][][] {
            {
                { -1.342, -0.447, 0.447, 1.342, 0.0, 0.0 }, // <START>
                { -1.342, -0.447, 0.447, 1.342, 0.0, 0.0 }  // <UNK>
            }
        }); // [1,2,6]
    
        // Comparer les logits réels avec les logits attendus
        double epsilon = 1e-3; // Tolérance ajustée
        boolean allMatch = true;
        StringBuilder mismatchDetails = new StringBuilder();
        for (int i = 0; i < expectedLogitsManualCorrect.size(0); i++) {
            for (int j = 0; j < expectedLogitsManualCorrect.size(1); j++) {
                for (int k = 0; k < expectedLogitsManualCorrect.size(2); k++) {
                    double expectedVal = expectedLogitsManualCorrect.getDouble(i, j, k);
                    double actualVal = logits.getDouble(i, j, k);
                    if (Math.abs(expectedVal - actualVal) > epsilon) {
                        allMatch = false;
                        mismatchDetails.append(String.format(
                            "Mismatch at (%d,%d,%d): expected %.6f but was %.6f%n",
                            i, j, k, expectedVal, actualVal));
                    }
                }
            }
        }
    
        if (!allMatch) {
            fail("Les logits de la cross-attention ne correspondent pas aux valeurs attendues:\n" + mismatchDetails.toString());
        }
    
        // Si tout correspond, afficher un message de succès
        System.out.println("Les poids de la cross-attention sont normalisés et les logits correspondent aux attentes connues avec une tolérance.");
    }


    @Test
    public void testBackwardPassLinearProjection() {
        // Initialiser LinearProjection avec LayerNorm désactivé
        LinearProjection lp = new LinearProjection(4, 6, false); // inputSize=4, outputSize=6, useLayerNorm=false
    
        // Initialiser les poids à une matrice identité étendue et biais à zéro
        INDArray identityWeights = NDArrayUtils.eye(4, 6); // [4,6]
        INDArray zeroBias = Nd4j.zeros(1, 6); // [1,6]
        lp.setWeights(identityWeights, zeroBias); // Utiliser setWeights de manière cohérente
    
        // Vérifier que les poids sont correctement définis
        System.out.println("Poids initiaux:");
        System.out.println(lp.getWeights());
        System.out.println("Biais initiaux:");
        System.out.println(lp.getBias());
    
        // Créer une entrée de rang 3 [batchSize=1, seqLength=1, inputSize=4]
        INDArray input = Nd4j.create(new float[][][] {
                {
                        { 1.0f, 2.0f, 3.0f, 4.0f }
                }
        }); // [1,1,4]
    
        // Définir une cible pour calculer la perte (par exemple, zeros)
        INDArray target = Nd4j.zeros(1, 1, 6); // [1,1,6]
    
        // Effectuer la projection
        INDArray output = lp.project(input); // [1,1,6]
    
        // Calculer la perte
        double loss = Transforms.pow(output.sub(target), 2).sumNumber().doubleValue() * 0.5f;
    
        // Calculer le gradient de la perte par rapport à l'output
        INDArray dLoss_dOutput = output.sub(target).mul(1.0f); // [1,1,6]
    
        // Effectuer la rétropropagation pour calculer les gradients analytiques
        Map<String, INDArray> gradients = lp.backward(input, dLoss_dOutput); // Map contenant les gradients
    
        INDArray dLoss_dWeights = gradients.get("weights"); // [4,6]
        INDArray dLoss_dBias = gradients.get("bias");       // [1,6]
    
        // Afficher les gradients analytiques
        System.out.println("Gradients analytiques dLoss/dWeights:");
        System.out.println(dLoss_dWeights);
        System.out.println("Gradients analytiques dLoss/dBias:");
        System.out.println(dLoss_dBias);
    
        // Calculer les gradients numériques via différences finies
        double epsilon = 1e-5;
        INDArray numericalGradWeights = Nd4j.zerosLike(identityWeights); // [4,6]
        INDArray numericalGradBias = Nd4j.zerosLike(zeroBias); // [1,6]
    
        // Calculer le gradient numérique pour chaque poids
        for (int i = 0; i < identityWeights.rows(); i++) {
            for (int j = 0; j < identityWeights.columns(); j++) {
                // Perturbation positive
                INDArray perturbedWeightsPlus = identityWeights.dup();
                perturbedWeightsPlus.putScalar(new int[]{i, j}, identityWeights.getDouble(i, j) + epsilon);
                lp.setWeights(perturbedWeightsPlus, zeroBias); // Mise à jour des poids
                System.out.println("Poids après perturbation positive (i=" + i + ", j=" + j + "):");
                System.out.println(lp.getWeights());
                INDArray outputPlus = lp.project(input);
                double lossPlus = Transforms.pow(outputPlus.sub(target), 2).sumNumber().doubleValue() * 0.5f;
    
                // Perturbation négative
                INDArray perturbedWeightsMinus = identityWeights.dup();
                perturbedWeightsMinus.putScalar(new int[]{i, j}, identityWeights.getDouble(i, j) - epsilon);
                lp.setWeights(perturbedWeightsMinus, zeroBias); // Mise à jour des poids
                System.out.println("Poids après perturbation négative (i=" + i + ", j=" + j + "):");
                System.out.println(lp.getWeights());
                INDArray outputMinus = lp.project(input);
                double lossMinus = Transforms.pow(outputMinus.sub(target), 2).sumNumber().doubleValue() * 0.5f;
    
                // Approximation du gradient
                double gradApprox = (lossPlus - lossMinus) / (2 * epsilon);
                numericalGradWeights.putScalar(new int[]{i, j}, gradApprox);
    
                // Remettre les poids à la valeur originale
                lp.setWeights(identityWeights, zeroBias); // Réinitialisation des poids
            }
        }
    
        // Calculer le gradient numérique pour chaque biais
        for (int j = 0; j < zeroBias.columns(); j++) {
            // Perturbation positive
            INDArray perturbedBiasPlus = zeroBias.dup();
            perturbedBiasPlus.putScalar(new int[]{0, j}, zeroBias.getDouble(0, j) + epsilon);
            lp.setWeights(identityWeights, perturbedBiasPlus); // Mise à jour du biais
            INDArray outputPlus = lp.project(input);
            double lossPlus = Transforms.pow(outputPlus.sub(target), 2).sumNumber().doubleValue() * 0.5f;
    
            // Perturbation négative
            INDArray perturbedBiasMinus = zeroBias.dup();
            perturbedBiasMinus.putScalar(new int[]{0, j}, zeroBias.getDouble(0, j) - epsilon);
            lp.setWeights(identityWeights, perturbedBiasMinus); // Mise à jour du biais
            INDArray outputMinus = lp.project(input);
            double lossMinus = Transforms.pow(outputMinus.sub(target), 2).sumNumber().doubleValue() * 0.5f;
    
            // Approximation du gradient
            double gradApprox = (lossPlus - lossMinus) / (2 * epsilon);
            numericalGradBias.putScalar(new int[]{0, j}, gradApprox);
    
            // Remettre les biais à la valeur originale
            lp.setWeights(identityWeights, zeroBias); // Réinitialisation du biais
        }
    
        // Afficher les gradients numériques
        System.out.println("Gradients numériques dLoss/dWeights:");
        System.out.println(numericalGradWeights);
        System.out.println("Gradients numériques dLoss/dBias:");
        System.out.println(numericalGradBias);
    
        // Comparer les gradients analytiques et numériques pour les poids
        double tolerance = 1e-3;
        boolean weightsMatch = true;
        StringBuilder mismatchDetailsWeights = new StringBuilder();
        for (int i = 0; i < 4; i++) { // dModel = 4
            for (int j = 0; j < 6; j++) { // outputSize = 6
                double expectedGrad = dLoss_dWeights.getDouble(i, j);
                double numericalGrad = numericalGradWeights.getDouble(i, j);
                if (Math.abs(expectedGrad - numericalGrad) > tolerance) {
                    weightsMatch = false;
                    mismatchDetailsWeights.append(String.format(
                            "Mismatch at weights (%d,%d): expected %.6f but was %.6f%n",
                            i, j, expectedGrad, numericalGrad));
                }
            }
        }
    
        // Comparer les gradients analytiques et numériques pour les biais
        boolean biasMatch = true;
        StringBuilder mismatchDetailsBias = new StringBuilder();
        for (int j = 0; j < 6; j++) { // outputSize = 6
            double expectedGrad = dLoss_dBias.getDouble(0, j);
            double numericalGrad = numericalGradBias.getDouble(0, j);
            if (Math.abs(expectedGrad - numericalGrad) > tolerance) {
                biasMatch = false;
                mismatchDetailsBias.append(String.format(
                        "Mismatch at bias (%d): expected %.6f but was %.6f%n",
                        j, expectedGrad, numericalGrad));
            }
        }
    
        // Vérifier les gradients des poids
        if (!weightsMatch) {
            fail("Les gradients des poids ne correspondent pas aux valeurs attendues:\n" + mismatchDetailsWeights.toString());
        }
    
        // Vérifier les gradients des biais
        if (!biasMatch) {
            fail("Les gradients des biais ne correspondent pas aux valeurs attendues:\n" + mismatchDetailsBias.toString());
        }
    
        // Si tout correspond, afficher un message de succès
        System.out.println("Les gradients analytiques et numériques de la projection linéaire correspondent avec une tolérance ajustée.");
    }


}
