package RN.transformer;


import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

public class calculateLearningRateOptimizerTest {

    private CustomAdamOptimizer optimizer;
    private final float initialLr = 0.001f;
    private final int warmupSteps = 1000;
    private final int modelSize = 512;
    private List<INDArray> parameters;
    private List<INDArray> gradients;

    @Before
    public void setUp() {
        parameters = new ArrayList<>();
        parameters.add(Nd4j.create(new float[]{0.1f, -0.2f}, new int[]{1, 2}));
        gradients = new ArrayList<>();
        gradients.add(Nd4j.create(new float[]{0.01f, -0.01f}, new int[]{1, 2}));
        
        optimizer = new CustomAdamOptimizer(initialLr, modelSize, warmupSteps, parameters);
    }

    @Test
    public void testCalculateLearningRateAtStart() {
        // Au début, le taux d'apprentissage devrait être initialLr
        assertEquals("At start, learning rate should be initialLr", initialLr, optimizer.getLearningRate(), 0.0);
    }

    @Test
    public void testCalculateLearningRateDuringWarmup() {

        // Simuler un pas pendant la période de warmup, par exemple à mi-chemin.
        optimizer.setCurrentStep(warmupSteps / 2);
        double lrMidWarmup = optimizer.calculateLearningRate();
        
        // Simuler le dernier pas de la période de warmup.
        optimizer.setCurrentStep(warmupSteps - 1);
        double lrEndWarmup = optimizer.calculateLearningRate();

        // Vérifier que le taux d'apprentissage à mi-chemin est supérieur à celui du départ.
        assertTrue("During warmup, learning rate should be more than initial at the start", lrMidWarmup > 0);

        // Vérifier que le taux d'apprentissage à la fin de la période de warmup ne dépasse pas initialLr.
        assertTrue("At the end of warmup, learning rate should not exceed initial learning rate", lrEndWarmup <= initialLr);

        // Optionnellement, vérifier que le taux d'apprentissage augmente au cours du warmup.
        assertTrue("Learning rate should increase during warmup", lrEndWarmup > lrMidWarmup);
    }

    @Test
    public void testCalculateLearningRateJustAfterWarmup() {
        // Juste après la période de warmup
        optimizer.setCurrentStep(warmupSteps + 1);
        double lrJustAfterWarmup = optimizer.calculateLearningRate();
        assertTrue("Just after warmup, learning rate should start to decrease", lrJustAfterWarmup < initialLr);
    }

    @Test
    public void testCalculateLearningRateWellAfterWarmup() {
        // Bien après la période de warmup
        optimizer.setCurrentStep(warmupSteps * 2);
        double lrWellAfterWarmup = optimizer.calculateLearningRate();
        assertTrue("Well after warmup, learning rate should be significantly lower than initialLr", lrWellAfterWarmup < initialLr);
    }
}
