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

    @Test
    public void testEndTokenCrossAttention() {
        
        // Initialiser les séquences source et cible avec <END>
        List<String> sourceSequence = Arrays.asList("<START>", "chien", "court", "dans", "le", "jardin", "<END>", "<PAD>", "<PAD>");
        List<String> targetSequence = Arrays.asList("<START>", "le", "chien", "court", "le", "jardin", "<END>", "<PAD>", "<PAD>");
        Batch batch = new Batch(sourceSequence, targetSequence, tokenizer);

        // Générer les masques
        INDArray keyPaddingMask = NDArrayUtils.createKeyPaddingMask(tokenizer, batch.getData()); // Séquence source
        INDArray queryPaddingMask = NDArrayUtils.createQueryPaddingMask(tokenizer, batch.getTarget()); // Séquence cible
        INDArray lookAheadMask = NDArrayUtils.createLookAheadMask(1, targetSequence.size());

        // Initialiser une instance simulée de l'encodeur si nécessaire
        Decoder encoder = null;

        // Initialiser une couche de décodeur avec l'encodeur
        DecoderLayer layer = new DecoderLayer(encoder, dModel, numHeads, dff, (double) dropout);

        // Exécuter la passe forward avec des entrées de rang 3
        INDArray x = Nd4j.randn(1, targetSequence.size(), dModel); // [1, 9, dModel]
        INDArray encoderOutput = Nd4j.randn(1, sourceSequence.size(), dModel); // [1, 9, dModel]
        INDArray output = layer.forward(true, x, encoderOutput, lookAheadMask, null, queryPaddingMask, keyPaddingMask, null, null);

        // Indice du token <END> dans la séquence cible
        int endTokenIdx = targetSequence.indexOf("<END>"); // Devrait être 6

        // Obtenir les poids d'attention pour <END>
        INDArray crossAttentionWeights = layer.encoderDecoderAttention.getAttentionWeights();
        
        // Vérifier les dimensions
        System.out.println("Shape de crossAttentionWeights: " + Arrays.toString(crossAttentionWeights.shape()));
        
        // Accéder aux poids d'attention du token <END>
        INDArray endAttentionWeightsArray = crossAttentionWeights.get(
            NDArrayIndex.point(0), // batch_index
            NDArrayIndex.point(0), // head_index
            NDArrayIndex.point(endTokenIdx), // query_token_index (<END>)
            NDArrayIndex.all() // key_token_indices
        );

        double[] endAttentionWeights = endAttentionWeightsArray.toDoubleVector();

        // Afficher les poids d'attention pour <END>
        System.out.println("Poids d'attention pour <END>: " + Arrays.toString(endAttentionWeights));

        // Vérifier que les poids d'attention pour <END> ne sont pas tous nuls
        double sumWeights = 0.0;
        for (int k = 0; k < sourceSequence.size(); k++) {
            sumWeights += crossAttentionWeights.getDouble(0, 0, endTokenIdx, k);
        }
        assertTrue("Les poids d'attention pour <END> ne devraient pas être tous nuls", sumWeights > 0.0);
    }

}
