package RN.transformer;


import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class calculateLearningRateOptimizerTest {

    private CustomAdamOptimizer optimizer;
    private final double initialLr = 0.001;
    private final int warmupSteps = 1000;
    private final long numberOfParameters = 100; // Supposons un certain nombre de paramètres pour l'initialisation
    private final int modelSize = 512;

    @Before
    public void setUp() {
        optimizer = new CustomAdamOptimizer(initialLr, modelSize, warmupSteps, numberOfParameters);
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
        optimizer.setCurrentStep(warmupSteps);
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
