package RN.transformer;

import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Gradient des POIDS (Wq, Wk, Wv, Wo) de l'attention par différences finies.
 * Les tests existants ne contrôlaient que le gradient par rapport aux ENTRÉES.
 * Perte = sum(sortie * G) avec G aléatoire fixe (évite les gradients dégénérés de sum(sortie)).
 */
public class MultiHeadAttentionWeightGradientCheckTest {

    private static final double EPS = 1e-2;

    private double worstError(String label, boolean cross) {
        Nd4j.getRandom().setSeed(5);
        int d = 8, heads = 2, tq = 4, tk = cross ? 6 : 4;
        MultiHeadAttention mha = new MultiHeadAttention(d, heads);
        INDArray q = Nd4j.randn(1, tq, d);
        INDArray kv = cross ? Nd4j.randn(1, tk, d) : q;
        INDArray qMask = Nd4j.ones(1, 1, tq, 1);
        INDArray kMask = Nd4j.ones(1, 1, 1, tk);
        INDArray look = null;
        INDArray g = Nd4j.randn(1, tq, d);

        java.util.function.Supplier<INDArray> fwd = () -> cross
                ? mha.forwardCrossAttention(q, kv, kv, qMask, kMask)
                : mha.forwardSelfAttention(q, kMask.reshape(1, 1, 1, tk), look);
        INDArray out = fwd.get();
        Map<String, INDArray> grads = mha.backward(g);
        String[] names = { "Wq", "Wk", "Wv", "Wo" };
        List<INDArray> params = mha.getParameters();
        double worst = 0;
        for (int w = 0; w < 4; w++) {
            INDArray an = grads.get(names[w]).dup();
            INDArray p = params.get(w);
            double maxErr = 0, maxAn = 0;
            for (int k = 0; k < p.length(); k++) {
                double orig = p.getDouble(k);
                p.putScalar(k, orig + EPS);
                double lp = fwd.get().mul(g).sumNumber().doubleValue();
                p.putScalar(k, orig - EPS);
                double lm = fwd.get().mul(g).sumNumber().doubleValue();
                p.putScalar(k, orig);
                double fd = (lp - lm) / (2 * EPS);
                maxErr = Math.max(maxErr, Math.abs(fd - an.reshape(-1).getDouble(k)));
                maxAn = Math.max(maxAn, Math.abs(an.reshape(-1).getDouble(k)));
            }
            System.out.printf("MHAW %s %s maxErr=%.5f (max|grad|=%.4f)%n", label, names[w], maxErr, maxAn);
            worst = Math.max(worst, maxErr);
        }
        return worst;
    }

    @Test
    public void testSelfAttentionWeightGradients() {
        assertTrue("self-attention : gradient des poids faux", worstError("self", false) < 2e-2);
    }

    @Test
    public void testCrossAttentionWeightGradients() {
        assertTrue("cross-attention : gradient des poids faux", worstError("cross", true) < 2e-2);
    }
}
