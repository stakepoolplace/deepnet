package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Persistance et entraînement CUMULATIF.
 *
 * Preuve centrale : entraîner 2 epochs, sauvegarder, recharger dans un modèle
 * NEUF, puis entraîner 2 epochs de plus doit donner les mêmes paramètres
 * qu'un entraînement continu de 4 epochs (dropout 0, données non mélangées).
 * Cela exige de persister aussi les moments d'Adam et le pas d'optimisation.
 */
public class TransformerPersistenceTest {

    private static final int MAX_LEN = 8;
    private static final int D = 16;
    private static final List<String> VOCAB = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>",
            "le", "chat", "chien", "mange", "dort", "sur", "tapis");
    private static final List<String> DATA = Arrays.asList("le chat mange", "le chien dort",
            "le chat dort sur le tapis", "le chien mange");

    private static TransformerModel build(List<String> vocab) {
        Nd4j.getRandom().setSeed(7);
        Tokenizer tok = new Tokenizer(vocab, D, MAX_LEN);
        TransformerModel m = new TransformerModel(1, D, 2, 32, 0.0, tok.getVocabSize(), tok, 0.01f, 2);
        m.getOptimizer().setMaxEpochs(1000);
        return m;
    }

    private static DataGenerator gen(TransformerModel m) {
        return new DataGenerator(DATA, DATA, m.tokenizer, 2, MAX_LEN);
    }

    private static void train(TransformerModel m, int epochs) {
        DataGenerator g = gen(m);
        for (int i = 0; i < epochs; i++) {
            m.train(g, 1);
        }
    }

    private static void assertSameParams(TransformerModel a, TransformerModel b, double tol, String msg) {
        List<INDArray> pa = a.getCombinedParameters();
        List<INDArray> pb = b.getCombinedParameters();
        assertEquals(pa.size(), pb.size());
        for (int i = 0; i < pa.size(); i++) {
            double diff = pa.get(i).sub(pb.get(i)).norm2Number().doubleValue();
            assertTrue(msg + " : paramètre " + i + " diffère (||Δ||=" + diff + ")", diff <= tol);
        }
    }

    private File tmp(String name) throws IOException {
        File f = File.createTempFile(name, ".ser");
        f.deleteOnExit();
        return f;
    }

    @Test
    public void testSaveLoadRoundTripKeepsParametersAndOptimizerState() throws Exception {
        TransformerModel a = build(VOCAB);
        train(a, 3);
        File f = tmp("roundtrip");
        a.saveState(f.getAbsolutePath());

        TransformerModel b = build(VOCAB);
        b.loadState(f.getAbsolutePath());

        assertSameParams(a, b, 0.0, "après rechargement");
        assertEquals("pas d'optimisation", a.getOptimizer().getCurrentStep(), b.getOptimizer().getCurrentStep());
        assertEquals("epoch", a.getOptimizer().getEpoch(), b.getOptimizer().getEpoch());
        assertTrue("isTrained restauré", b.isTrained());
        assertEquals("inférence identique", a.generate("le chat", 5, 0.05, 1, 1L), b.generate("le chat", 5, 0.05, 1, 1L));
    }

    /** Précondition de la preuve de reprise : deux entraînements identiques doivent coïncider. */
    @Test
    public void testTrainingIsDeterministic() {
        TransformerModel a = build(VOCAB);
        train(a, 4);
        TransformerModel b = build(VOCAB);
        train(b, 4);
        assertSameParams(a, b, 1e-3, "deux entraînements identiques");
    }

    @Test
    public void testResumedTrainingEqualsContinuousTraining() throws Exception {
        TransformerModel continuous = build(VOCAB);
        train(continuous, 4);

        TransformerModel first = build(VOCAB);
        train(first, 2);
        File f = tmp("resume");
        first.saveState(f.getAbsolutePath());

        TransformerModel resumed = build(VOCAB);
        resumed.loadState(f.getAbsolutePath());
        train(resumed, 2);

        assertEquals(continuous.getOptimizer().getCurrentStep(), resumed.getOptimizer().getCurrentStep());
        assertSameParams(continuous, resumed, 1e-3, "reprise vs entraînement continu");
    }

    @Test
    public void testLoadRejectsIncompatibleVocabulary() throws Exception {
        TransformerModel a = build(VOCAB);
        train(a, 1);
        File f = tmp("vocab");
        a.saveState(f.getAbsolutePath());

        List<String> other = new ArrayList<>(VOCAB);
        other.set(5, "oiseau"); // même taille, mot différent
        TransformerModel b = build(other);
        try {
            b.loadState(f.getAbsolutePath());
            fail("Un vocabulaire différent doit être refusé");
        } catch (IOException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("vocabulaire"));
        }
    }
}
