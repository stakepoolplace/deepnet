package RN.transformer;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import RN.transformer.Decoder.DecoderLayer;

import java.util.Arrays;
import java.util.List;

public class DecoderTest {

    private Decoder decoder;
    private Tokenizer tokenizer;
    private INDArray encodedInput;
    private INDArray encoderInputTokens;

    @Before
    public void setUp() {
        // Initialisation du Tokenizer avec un vocabulaire simple
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        int dModel = 64;
        int maxSequenceLength = 3;
        tokenizer = new Tokenizer(vocabulary, dModel, maxSequenceLength);

        // Initialisation du décodeur
        int numLayers = 2;
        int numHeads = 1;
        int dff = 128;
        double dropoutRate = 0.0;
        decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, tokenizer, true);

        // Simuler une entrée encodée
        encodedInput = Nd4j.rand(new int[]{1, maxSequenceLength, dModel});
        encoderInputTokens = Nd4j.create(new int[][]{{tokenizer.getStartTokenId(), tokenizer.getEndTokenId(), tokenizer.getPadTokenId()}});
    }

    @Test
    public void testDecoderFunctionality() {

        // Initialiser les IDs de sortie avec le token de début
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getPadTokenId(), tokenizer.getEndTokenId());

        // Convertir les IDs de sortie en INDArray 2D
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });

        // Créer un Batch pour le décodeur
        Batch decoderBatch = new Batch(decoderInputIds, null, tokenizer);
        
        // Décoder en passant les tokens d'entrée de l'encodeur
        INDArray logits = decoder.decode(false, encodedInput, encodedInput, decoderBatch, encoderInputTokens);

        // Vérifier que les logits ne sont pas nuls
        assertNotNull("Les logits ne devraient pas être null", logits);

        // Vérifier la forme des logits
        assertEquals("La forme des logits devrait être [1, seqLength, vocabSize]",
                3, logits.shape().length);

    }

    @Test
    public void testAttentionMechanism() {
        // Initialiser les IDs de sortie avec une séquence valide
        List<Integer> outputIds = Arrays.asList(tokenizer.getStartTokenId(), tokenizer.getUnkTokenId(), tokenizer.getEndTokenId());

        // Convertir les IDs de sortie en INDArray 2D avec la bonne forme [1, seqLength]
        INDArray decoderInputIds = Nd4j.create(new int[][] { outputIds.stream().mapToInt(i -> i).toArray() });

        // Créer un Batch pour le décodeur avec les tokens d'entrée
        Batch decoderBatch = new Batch(decoderInputIds, decoderInputIds, tokenizer);

        // Décoder en passant les tokens d'entrée de l'encodeur
        INDArray logits = decoder.decode(true, encodedInput, encodedInput, decoderBatch, encoderInputTokens);

        // Tracer les valeurs intermédiaires
        System.out.println("Logits: " + logits);

        // Vérifier que les logits ne sont pas nuls
        assertNotNull("Les logits ne devraient pas être null", logits);

        // Vérifier la forme des logits
        assertEquals("La forme des logits devrait être [1, seqLength, vocabSize]",
                3, logits.shape().length);

        // Tracer les poids d'attention
        for (int i = 0; i < decoder.layers.size(); i++) {
            DecoderLayer layer = decoder.layers.get(i);
            MultiHeadAttention selfAttn = layer.getSelfAttention();
            MultiHeadAttention crossAttn = layer.getCrossAttention();

            System.out.println("===== Decoder Layer " + (i + 1) + " Self-Attention Weights =====");
            selfAttn.printAttentionWeights(tokenizer.idsToListTokens(outputIds), 
                                         tokenizer.idsToListTokens(outputIds), 
                                         0, 
                                         tokenizer.getIdToTokenMap());

            System.out.println("===== Decoder Layer " + (i + 1) + " Cross-Attention Weights =====");
            crossAttn.printAttentionWeights(tokenizer.idsToListTokens(outputIds),
                                          Arrays.asList("<START>", "<PAD>", "<END>"),
                                          0,
                                          tokenizer.getIdToTokenMap());
        }
    }







}
