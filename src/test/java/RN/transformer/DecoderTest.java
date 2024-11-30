package RN.transformer;

import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.dropout;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import RN.transformer.Decoder.DecoderLayer;
import RN.utils.NDArrayUtils;

public class DecoderTest {

    private static TransformerModel transformerModel;
    private static int dModel;
    private static int maxSeqLength;
    private static int numHeads;
    private static int dff;
    private static float dropout;
    private static int vocabSize;
    // Initialisation du Tokenizer
    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() {
        // Initialisation des paramètres
        dModel = 300;
        numHeads = 6;
        maxSeqLength = 11;
        dff = 512;
        dropout = 0.0f;
        List<String> vocab = Arrays.asList("<PAD>", "<START>", "<END>", "chat", "mange", "souris", "la");
        vocabSize = vocab.size();
        tokenizer = new Tokenizer(vocab, dModel, maxSeqLength);
    }



}
