package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/** Briques de la recette d'entraînement : clipping, planning de LR, early stopping, perte de validation. */
public class TrainingRecipeTest {

    private static List<INDArray> grads() {
        return new ArrayList<>(Arrays.asList(Nd4j.create(new float[] { 3f, 0f }), Nd4j.create(new float[] { 0f, 4f })));
    }

    @Test
    public void testGlobalNormClippingPreservesDirection() {
        List<INDArray> g = grads();
        double before = TransformerModel.clipGradients(g, TransformerModel.ClipMode.GLOBAL_NORM, 1.0);
        assertEquals(5.0, before, 1e-6);
        assertEquals(0.6, g.get(0).getDouble(0), 1e-6);
        assertEquals(0.8, g.get(1).getDouble(1), 1e-6);
    }

    @Test
    public void testPerTensorClippingIsTheHistoricalBehaviour() {
        List<INDArray> g = grads();
        TransformerModel.clipGradients(g, TransformerModel.ClipMode.PER_TENSOR, 1.0);
        assertEquals(1.0, g.get(0).norm2Number().doubleValue(), 1e-6);
        assertEquals(1.0, g.get(1).norm2Number().doubleValue(), 1e-6);
    }

    @Test
    public void testNoClippingBelowThreshold() {
        List<INDArray> g = grads();
        TransformerModel.clipGradients(g, TransformerModel.ClipMode.GLOBAL_NORM, 10.0);
        assertEquals(3.0, g.get(0).getDouble(0), 1e-6);
        assertEquals(4.0, g.get(1).getDouble(1), 1e-6);
    }

    @Test
    public void testCosineScheduleValues() {
        CustomAdamOptimizer o = new CustomAdamOptimizer(1e-3f, 16, 50, new ArrayList<>());
        o.configureSchedule(CustomAdamOptimizer.Schedule.COSINE, 1e-3f, 10, 110, 0.1f);
        assertEquals(1e-4f, o.learningRateAt(0), 1e-9f);          // début du warmup
        assertEquals(1e-3f, o.learningRateAt(9), 1e-9f);          // fin du warmup = pic
        assertEquals(5.5e-4f, o.learningRateAt(60), 1e-6f);       // milieu de décroissance
        assertEquals(1e-4f, o.learningRateAt(110), 1e-9f);        // plancher = 10 % du pic
        assertEquals(1e-4f, o.learningRateAt(5000), 1e-9f);       // reste au plancher
    }

    @Test
    public void testScheduleRestartsFromCurrentStepOnResume() {
        CustomAdamOptimizer o = new CustomAdamOptimizer(1e-3f, 16, 50, new ArrayList<>());
        o.setCurrentStep(12000); // checkpoint déjà très avancé
        o.configureSchedule(CustomAdamOptimizer.Schedule.COSINE, 2e-4f, 5, 100, 0.05f);
        assertEquals(2e-4f, o.learningRateAt(12004), 1e-9f);      // pic atteint 5 pas après la reprise
        assertEquals(1e-5f, o.learningRateAt(12100), 1e-9f);
    }

    @Test
    public void testLinearAndConstantSchedules() {
        CustomAdamOptimizer o = new CustomAdamOptimizer(1e-3f, 16, 50, new ArrayList<>());
        o.configureSchedule(CustomAdamOptimizer.Schedule.LINEAR, 1e-3f, 0, 100, 0f);
        assertEquals(5e-4f, o.learningRateAt(50), 1e-9f);
        o.configureSchedule(CustomAdamOptimizer.Schedule.CONSTANT, 3e-4f, 0, 100, 0f);
        assertEquals(3e-4f, o.learningRateAt(77), 1e-9f);
    }

    @Test
    public void testEarlyStopping() {
        EarlyStopping es = new EarlyStopping(2, 0.01);
        assertTrue(es.update(1.0, 1));
        assertTrue(es.update(0.9, 2));
        assertFalse("amélioration < minDelta ignorée", es.update(0.895, 3));
        assertFalse(es.shouldStop());
        assertFalse(es.update(Double.NaN, 4));
        assertTrue(es.shouldStop());
        assertEquals(0.9, es.getBest(), 1e-12);
        assertEquals(2, es.getBestStep());
    }

    @Test
    public void testEvaluateLossIsPureAndDeterministic() {
        List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "le", "chat", "chien", "mange", "dort");
        Nd4j.getRandom().setSeed(5);
        Tokenizer tok = new Tokenizer(vocab, 16, 8);
        TransformerModel m = new TransformerModel(1, 16, 2, 32, 0.0, tok.getVocabSize(), tok, 0.01f, 2);
        DataGenerator g = new DataGenerator(Arrays.asList("le chat", "le chien"), Arrays.asList("mange", "dort"), tok, 2, 8);
        List<INDArray> before = new ArrayList<>();
        for (INDArray p : m.getCombinedParameters()) before.add(p.dup());
        float a = m.evaluateLoss(g);
        float b = m.evaluateLoss(g);
        assertEquals(a, b, 1e-6f);
        assertTrue(a > 0 && !Float.isNaN(a));
        for (int i = 0; i < before.size(); i++) {
            assertTrue("evaluateLoss ne doit pas modifier les poids", before.get(i).equals(m.getCombinedParameters().get(i)));
        }
        m.setTrained(true);
        for (int i = 0; i < 30; i++) m.train(g, 1);
        assertTrue("la perte de validation baisse après entraînement", m.evaluateLoss(g) < a);
    }
}
