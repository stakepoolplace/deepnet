package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Vérification de bout en bout du gradient du MODÈLE COMPLET par différences finies :
 * perte rapportée (moyenne par token) -> backward -> gradients de TOUS les paramètres
 * (embeddings, attention, FFN, LayerNorm, projection).
 *
 * Établit en particulier que, en mode SUM (historique), le gradient est exactement
 * N fois celui de la perte rapportée (N = tokens supervisés), et qu'en mode MEAN_PER_TOKEN
 * il lui est égal.
 */
public class TransformerEndToEndGradientCheckTest {

    private static final List<String> VOCAB = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>",
            "le", "chat", "chien", "mange", "dort", "sur");
    private static final List<String> IN = Arrays.asList("le chat", "le chien dort");
    private static final List<String> OUT = Arrays.asList("mange sur", "dort");

    private static class Setup {
        TransformerModel model;
        DataGenerator gen;
        int validTokens;
    }

    private static Setup build(TransformerModel.LossNormalization norm) {
        Nd4j.getRandom().setSeed(11);
        Tokenizer tok = new Tokenizer(VOCAB, 8, 6);
        Setup s = new Setup();
        s.model = new TransformerModel(1, 8, 2, 16, 0.0, tok.getVocabSize(), tok, 0.001f, 1);
        s.model.setLossNormalization(norm);
        // LR nul + clipping neutre : train() calcule et laisse les gradients sans modifier les poids.
        s.model.setGradientClipping(TransformerModel.ClipMode.GLOBAL_NORM, 1e18);
        s.model.getOptimizer().configureSchedule(CustomAdamOptimizer.Schedule.CONSTANT, 0f, 0, 1, 0f);
        s.gen = new DataGenerator(IN, OUT, tok, 2, 6);
        // tokens supervisés = mots de la suite + END
        s.validTokens = 0;
        for (String o : OUT) s.validTokens += o.split(" ").length + 1;
        return s;
    }

    private static List<INDArray> analyticGradients(Setup s) {
        s.model.train(s.gen, 1);
        List<INDArray> out = new ArrayList<>();
        for (INDArray g : s.model.getCombinedGradients()) out.add(g.dup());
        return out;
    }

    @Test
    public void testFiniteDifferencesMatchMeanPerTokenGradient() {
        finiteDifferenceCheck(build(TransformerModel.LossNormalization.MEAN_PER_TOKEN), 0.0);
    }

    /** Même contrôle APRÈS perturbation de tous les paramètres (gamma != 1, beta != 0 : cas que l'initialisation masque). */
    @Test
    public void testFiniteDifferencesAfterRandomizingAllParameters() {
        finiteDifferenceCheck(build(TransformerModel.LossNormalization.MEAN_PER_TOKEN), 0.15);
    }

    private void finiteDifferenceCheck(Setup s, double noise) {
        if (noise > 0) {
            Nd4j.getRandom().setSeed(99);
            for (INDArray q : s.model.getCombinedParameters()) {
                q.addi(Nd4j.randn(q.shape()).mul(noise));
            }
        }
        List<INDArray> analytic = analyticGradients(s);
        List<INDArray> params = s.model.getCombinedParameters();
        assertEquals(params.size(), analytic.size());

        final double h = 1e-2;
        Random rnd = new Random(3);
        int checked = 0, failed = 0;
        StringBuilder bad = new StringBuilder();
        for (int i = 0; i < params.size(); i++) {
            INDArray p = params.get(i);
            long n = p.length();
            // les 2 éléments de plus fort gradient + 2 aléatoires
            List<Long> idx = new ArrayList<>();
            INDArray flatG = analytic.get(i).reshape(-1);
            long best = 0, second = -1;
            for (long k = 0; k < n; k++) {
                if (Math.abs(flatG.getDouble(k)) > Math.abs(flatG.getDouble(best))) best = k;
            }
            idx.add(best);
            for (int r = 0; r < 2; r++) idx.add((long) rnd.nextInt((int) n));
            for (long k : idx) {
                double orig = p.getDouble(k);
                p.putScalar(k, orig + h);
                double lp = s.model.evaluateLoss(s.gen);
                p.putScalar(k, orig - h);
                double lm = s.model.evaluateLoss(s.gen);
                p.putScalar(k, orig);
                double fd = (lp - lm) / (2 * h);
                double an = flatG.getDouble(k);
                double err = Math.abs(fd - an);
                boolean ok = err <= 2e-3 + 0.05 * Math.abs(an);
                checked++;
                if (!ok) {
                    failed++;
                    bad.append(String.format("param#%d shape=%s idx=%d analytic=%.6f fd=%.6f%n", i,
                            Arrays.toString(p.shape()), k, an, fd));
                }
            }
        }
        System.out.println("GRADCHECK vérifiés=" + checked + " échecs=" + failed + "\n" + bad);
        assertTrue("Gradient analytique ≠ différences finies :\n" + bad, failed == 0);
    }

    @Test
    public void testSumModeIsExactlyNTimesTheMeanGradient() {
        Setup mean = build(TransformerModel.LossNormalization.MEAN_PER_TOKEN);
        List<INDArray> gMean = analyticGradients(mean);
        Setup sum = build(TransformerModel.LossNormalization.SUM);
        List<INDArray> gSum = analyticGradients(sum);
        double ratioSum = 0;
        int n = 0;
        for (int i = 0; i < gMean.size(); i++) {
            double nm = gMean.get(i).norm2Number().doubleValue();
            double ns = gSum.get(i).norm2Number().doubleValue();
            if (nm > 1e-8) {
                assertEquals("paramètre " + i, mean.validTokens, ns / nm, 1e-3 * mean.validTokens);
                n++;
            }
        }
        assertTrue(n > 5);
    }

    @Test
    public void testTrainLossEqualsEvaluateLoss() {
        Setup s = build(TransformerModel.LossNormalization.MEAN_PER_TOKEN);
        // un seul batch (2 paires, batch=2) : la perte d'entraînement = perte de validation même poids
        float train = s.model.train(s.gen, 1);
        float eval = s.model.evaluateLoss(s.gen);
        assertEquals(train, eval, 1e-4f);
    }
}
