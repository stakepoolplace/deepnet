package RN.transformer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import RN.transformer.Decoder.DecoderLayer;
import RN.utils.NDArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Garde-fou de la rétropropagation : compare les gradients analytiques aux
 * différences finies (float32, eps=1e-3). Toute régression du backward
 * (connexions résiduelles, routage cross-attention, LayerNorm...) fait
 * échouer ce test avec une erreur d'ordre O(0.1-1) contre ~1e-4 attendus.
 */
public class TransformerGradientCheckTest {

    private static final double EPS = 1e-3;
    private static final double TOLERANCE = 1e-2;

    private static double check(String name, Function<INDArray, INDArray> forward,
                                Function<INDArray, INDArray> backward, INDArray x) {
        INDArray out = forward.apply(x);
        INDArray analytic = backward.apply(Nd4j.onesLike(out));
        long seqLen = x.shape()[1];
        long dModel = x.shape()[2];
        double maxErr = 0;
        for (int j = 0; j < seqLen; j++) {
            for (int k = 0; k < dModel; k++) {
                INDArray xp = x.dup();
                xp.putScalar(new int[] { 0, j, k }, xp.getDouble(0, j, k) + EPS);
                INDArray xm = x.dup();
                xm.putScalar(new int[] { 0, j, k }, xm.getDouble(0, j, k) - EPS);
                double num = (forward.apply(xp).sumNumber().doubleValue()
                        - forward.apply(xm).sumNumber().doubleValue()) / (2 * EPS);
                maxErr = Math.max(maxErr, Math.abs(num - analytic.getDouble(0, j, k)));
            }
        }
        System.out.println("Gradient check " + name + " maxErr=" + maxErr);
        return maxErr;
    }

    @Test
    public void gradientCheckSubModules() {
        Nd4j.getRandom().setSeed(7);

        LayerNorm ln = new LayerNorm(4);
        INDArray xln = Nd4j.randn(1, 3, 4);
        double errLN = check("LayerNorm", ln::forward, g -> ln.backward(g).get("input"), xln);

        PositionwiseFeedForward ff = new PositionwiseFeedForward(4, 8);
        INDArray xff = Nd4j.randn(1, 3, 4);
        double errFF = check("FeedForward", ff::forward, g -> ff.backward(g).get("input"), xff);

        // En self-attention x alimente Q, K et V : dL/dx = somme des trois
        MultiHeadAttention mha = new MultiHeadAttention(4, 1);
        INDArray xmha = Nd4j.randn(1, 3, 4);
        double errSelf = check("MHA-self", v -> mha.forwardSelfAttention(v, null, null), g -> {
            Map<String, INDArray> gm = mha.backward(g);
            return gm.get("gradInputQ").add(gm.get("gradInputK")).add(gm.get("gradInputV"));
        }, xmha);

        // En cross-attention seule la requête varie ici : dL/dQ
        MultiHeadAttention mhac = new MultiHeadAttention(4, 1);
        INDArray q = Nd4j.randn(1, 3, 4);
        INDArray kv = Nd4j.randn(1, 3, 4);
        double errCross = check("MHA-cross-Q", v -> mhac.forwardCrossAttention(v, kv, kv, null, null),
                g -> mhac.backward(g).get("gradInputQ"), q);

        // Gradients des poids : différences finies sur chaque coefficient
        MultiHeadAttention mhaw = new MultiHeadAttention(4, 1);
        INDArray xw = Nd4j.randn(1, 3, 4);
        mhaw.forwardSelfAttention(xw, null, null);
        Map<String, INDArray> gw = mhaw.backward(Nd4j.ones(1, 3, 4));
        double errW = 0;
        for (String key : new String[] { "Wq", "Wk", "Wv", "Wo" }) {
            INDArray w = null;
            if (key.equals("Wq")) w = mhaw.getWq();
            if (key.equals("Wk")) w = mhaw.getWk();
            if (key.equals("Wv")) w = mhaw.getWv();
            if (key.equals("Wo")) w = mhaw.getWo();
            double errKey = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    INDArray wOrig = w.dup();
                    w.putScalar(new int[] { i, j }, wOrig.getDouble(i, j) + EPS);
                    double lp = mhaw.forwardSelfAttention(xw, null, null).sumNumber().doubleValue();
                    w.putScalar(new int[] { i, j }, wOrig.getDouble(i, j) - EPS);
                    double lm = mhaw.forwardSelfAttention(xw, null, null).sumNumber().doubleValue();
                    w.assign(wOrig);
                    double num = (lp - lm) / (2 * EPS);
                    errKey = Math.max(errKey, Math.abs(num - gw.get(key).getDouble(i, j)));
                }
            }
            System.out.println("Gradient check MHA-weights " + key + " maxErr=" + errKey);
            errW = Math.max(errW, errKey);
        }
        System.out.println("Gradient check MHA-weights maxErr=" + errW);

        // Cross-attention à longueurs inégales (cas réel : Tq=2 requêtes, Tk=5 clés) :
        // dL/dK et dL/dV doivent correspondre aux différences finies.
        MultiHeadAttention mhau = new MultiHeadAttention(4, 1);
        INDArray qu = Nd4j.randn(1, 2, 4);
        INDArray kvu = Nd4j.randn(1, 5, 4);
        INDArray outu = mhau.forwardCrossAttention(qu, kvu, kvu, null, null);
        final Map<String, INDArray> gu = mhau.backward(Nd4j.onesLike(outu));
        double errCrossKV = 0;
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 4; k++) {
                INDArray kvp = kvu.dup();
                kvp.putScalar(new int[] { 0, j, k }, kvp.getDouble(0, j, k) + EPS);
                INDArray kvm = kvu.dup();
                kvm.putScalar(new int[] { 0, j, k }, kvm.getDouble(0, j, k) - EPS);
                double num = (mhau.forwardCrossAttention(qu, kvp, kvp, null, null).sumNumber().doubleValue()
                        - mhau.forwardCrossAttention(qu, kvm, kvm, null, null).sumNumber().doubleValue())
                        / (2 * EPS);
                double ana = gu.get("gradInputK").getDouble(0, j, k) + gu.get("gradInputV").getDouble(0, j, k);
                errCrossKV = Math.max(errCrossKV, Math.abs(num - ana));
            }
        }
        System.out.println("Gradient check MHA-cross-KV-unbalanced maxErr=" + errCrossKV);

        if (errLN >= TOLERANCE || errFF >= TOLERANCE || errSelf >= TOLERANCE || errCross >= TOLERANCE
                || errW >= TOLERANCE || errCrossKV >= TOLERANCE) {
            throw new AssertionError("Échec du gradient check (voir maxErr ci-dessus)");
        }
    }

    @Test
    public void gradientCheckDecoderLayer() {        Nd4j.getRandom().setSeed(7);
        int dModel = 4;
        List<String> vocabulary = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>");
        Tokenizer tokenizer = new Tokenizer(vocabulary, dModel, 3);
        Decoder decoder = new Decoder(1, dModel, 1, 8, 0.0, tokenizer, true);
        DecoderLayer layer = decoder.layers.get(0);
        INDArray x = Nd4j.randn(1, 3, dModel);
        INDArray enc = Nd4j.randn(1, 3, dModel);

        double errX = check("DecoderLayer-dx",
                v -> layer.forward(false, v, enc, null, null, null, null, null, null), g -> {
                    Map<String, INDArray> gradOut = new HashMap<>();
                    gradOut.put("input", g);
                    return layer.backward(gradOut, null).get("input");
                }, x);

        if (errX >= TOLERANCE) {
            throw new AssertionError("Échec du gradient check DecoderLayer-dx, maxErr=" + errX);
        }

        // Chemin vers l'encodeur : dL/d(enc) doit valoir dL/dK + dL/dV (cross-attention)
        INDArray out = layer.forward(false, x, enc, null, null, null, null, null, null);
        Map<String, INDArray> gradOut = new HashMap<>();
        gradOut.put("input", Nd4j.onesLike(out));
        final Map<String, INDArray> grads = layer.backward(gradOut, null);
        INDArray analyticEnc = grads.get("gradInputK").add(grads.get("gradInputV"));
        double errEnc = 0;
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < dModel; k++) {
                INDArray encp = enc.dup();
                encp.putScalar(new int[] { 0, j, k }, encp.getDouble(0, j, k) + EPS);
                INDArray encm = enc.dup();
                encm.putScalar(new int[] { 0, j, k }, encm.getDouble(0, j, k) - EPS);
                double num = (layer.forward(false, x, encp, null, null, null, null, null, null).sumNumber().doubleValue()
                        - layer.forward(false, x, encm, null, null, null, null, null, null).sumNumber().doubleValue())
                        / (2 * EPS);
                errEnc = Math.max(errEnc, Math.abs(num - analyticEnc.getDouble(0, j, k)));
            }
        }
        System.out.println("Gradient check DecoderLayer-dEnc maxErr=" + errEnc);
        if (errEnc >= TOLERANCE) {
            throw new AssertionError("Échec du gradient check DecoderLayer-dEnc, maxErr=" + errEnc);
        }
    }

    @Test
    public void gradientCheckMultiHeadMasked() {
        // Cas réel : plusieurs têtes + masques de padding + look-ahead.
        // Les permutations/reshape dépendants de H ne sont testés qu'ici.
        Nd4j.getRandom().setSeed(21);
        int dModel = 8;
        int numHeads = 2;
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        INDArray q = Nd4j.randn(1, 4, dModel);
        INDArray kv = Nd4j.randn(1, 6, dModel);
        INDArray qMask = Nd4j.ones(1, 1, 4, 1);
        INDArray kMask = Nd4j.ones(1, 1, 1, 6);
        // Masquer la dernière clé comme du padding
        kMask.putScalar(new int[] { 0, 0, 0, 5 }, 0.0);
        INDArray out = mha.forwardCrossAttention(q, kv, kv, qMask, kMask);
        // IMPORTANT : backward IMMÉDIATEMENT, avant tout autre forward qui
        // écraserait les caches Q/K/V/attention utilisés par backward.
        Map<String, INDArray> g = mha.backward(Nd4j.onesLike(out));
        INDArray analyticQ = g.get("gradInputQ").dup();
        INDArray analyticKV = g.get("gradInputK").add(g.get("gradInputV")).dup();

        // dL/dQ (cross) par différences finies (Kv fixé)
        double maxErr = 0;
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < dModel; k++) {
                INDArray qp = q.dup();
                qp.putScalar(new int[] { 0, j, k }, qp.getDouble(0, j, k) + EPS);
                INDArray qm = q.dup();
                qm.putScalar(new int[] { 0, j, k }, qm.getDouble(0, j, k) - EPS);
                double num = (mha.forwardCrossAttention(qp, kv, kv, qMask, kMask).sumNumber().doubleValue()
                        - mha.forwardCrossAttention(qm, kv, kv, qMask, kMask).sumNumber().doubleValue()) / (2 * EPS);
                maxErr = Math.max(maxErr, Math.abs(num - analyticQ.getDouble(0, j, k)));
            }
        }
        System.out.println("Gradient check MHA-multihead-masked-dQ maxErr=" + maxErr);

        // dL/dK+dL/dV (cross) par différences finies (Q fixée) : utilise les
        // poids analytiques déjà extraits (les forwards ci-dessous écrasent les caches)
        double maxErrKV = 0;
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < dModel; k++) {
                INDArray kvp = kv.dup();
                kvp.putScalar(new int[] { 0, j, k }, kvp.getDouble(0, j, k) + EPS);
                INDArray kvm = kv.dup();
                kvm.putScalar(new int[] { 0, j, k }, kvm.getDouble(0, j, k) - EPS);
                double num = (mha.forwardCrossAttention(q, kvp, kvp, qMask, kMask).sumNumber().doubleValue()
                        - mha.forwardCrossAttention(q, kvm, kvm, qMask, kMask).sumNumber().doubleValue()) / (2 * EPS);
                maxErrKV = Math.max(maxErrKV, Math.abs(num - analyticKV.getDouble(0, j, k)));
            }
        }
        System.out.println("Gradient check MHA-multihead-masked-dKV maxErr=" + maxErrKV);

        if (maxErr >= TOLERANCE || maxErrKV >= TOLERANCE) {
            throw new AssertionError("Échec du gradient check multi-têtes masqué");
        }
    }
}
