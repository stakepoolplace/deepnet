package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/** Le corpus v2 s'enchaîne de bout en bout : tokenizer -> entraînement dynamique -> évaluation à 3 niveaux. */
public class CorpusV2PipelineSmokeTest {

    @Test
    public void testTrainOneEpochAndEvaluateOnThreeLevels() {
        Nd4j.getRandom().setSeed(42);
        Tokenizer tok = new Tokenizer(CorpusV2.vocabulary(), 64, CorpusV2.MAX_LEN);
        TransformerModel m = new TransformerModel(2, 64, 4, 256, 0.0, tok.getVocabSize(), tok, 5e-4f, 50);
        m.setLossNormalization(TransformerModel.LossNormalization.MEAN_PER_TOKEN);
        m.setGradientClipping(TransformerModel.ClipMode.GLOBAL_NORM, 1.0);

        NextWordProtocol.Sample s = CorpusV2.dynamicSample(42, 1, 200, 3, CorpusV2.forbiddenForTraining());
        List<String> in = new ArrayList<>(), out = new ArrayList<>();
        for (NextWordProtocol.Item it : s.items) { in.add(it.prefix); out.add(it.continuation); }
        float loss = m.train(new DataGenerator(in, out, tok, 16, CorpusV2.MAX_LEN), 1);
        assertTrue("perte finie", !Float.isNaN(loss) && !Float.isInfinite(loss) && loss > 0);

        List<NextWordProtocol.Item> val = CorpusV2.validationItems().subList(0, 30);
        CorpusV2.EvalV2 r = CorpusV2.evaluate(m, tok, val, "smoke");
        System.out.println("CORPUSV2SMOKE " + r);
        assertEquals(30, r.n);
        assertTrue("niveaux cumulatifs N3 <= N2 <= N1 <= n", r.l3 <= r.l2 && r.l2 <= r.l1 && r.l1 <= r.n);
        assertTrue(Double.isFinite(r.nll) && Double.isFinite(r.oracleNll) && r.oracleNll > 0);
        assertTrue("plafond d'exactitude dans (0,1]", r.modeCeiling > 0 && r.modeCeiling <= 1);
    }
}
