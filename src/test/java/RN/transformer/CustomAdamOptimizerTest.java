package RN.transformer;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CustomAdamOptimizerTest {

    private CustomAdamOptimizer optimizer;
    private List<INDArray> parameters;
    private List<INDArray> gradients;
    private final double initialLr = 0.001;
    private final int warmupSteps = 1000;
    

    @Before
    public void setUp() {
        // Initialiser directement le nombre total de paramètres basé sur ce que vous allez ajouter à 'parameters'
        int numberOfParameters = 2; // Par exemple, si vous savez que vous ajouterez un INDArray avec 2 éléments

        optimizer = new CustomAdamOptimizer(initialLr, warmupSteps, numberOfParameters);
        
        // Ensuite, initialiser 'parameters' avec les valeurs de test
        parameters = new ArrayList<>();
        parameters.add(Nd4j.create(new float[]{0.1f, -0.2f}, new int[]{1, 2}));
        gradients = new ArrayList<>();
        gradients.add(Nd4j.create(new float[]{0.01f, -0.01f}, new int[]{1, 2}));
    }


    @Test
    public void testUpdate() {
        INDArray originalParams = parameters.get(0).dup();
        optimizer.update(parameters, gradients);
        INDArray updatedParams = parameters.get(0);

        Assert.assertTrue("Parameters should be updated after optimizer update call",
                !originalParams.equalsWithEps(updatedParams, 1e-7)); // Utiliser equalsWithEps pour une comparaison avec une tolérance

    }

    @Test
    public void testLearningRateAdjustment() {
        // Simulate a few steps to trigger learning rate adjustments
        for (int i = 0; i < warmupSteps / 2; i++) {
            optimizer.update(parameters, gradients);
        }

        double learningRateMidway = optimizer.getLearningRate();
        
        // Verify that learning rate during warmup is less than initial and decreases with steps
        Assert.assertTrue("Learning rate should increase during warmup",
                          learningRateMidway > 0 && learningRateMidway <= initialLr);

        // Simulate more steps to complete warmup
        for (int i = warmupSteps / 2; i < warmupSteps * 2; i++) {
            optimizer.update(parameters, gradients);
        }

        double learningRateAfterWarmup = optimizer.getLearningRate();

        // Verify learning rate after warmup is less than during warmup
        Assert.assertTrue("Learning rate should decrease after warmup",
                          learningRateAfterWarmup < initialLr);
    }
    
    @Test
    public void testLearningRateSetter() {
        double newLearningRate = 0.0001;
        optimizer.setLearningRate(newLearningRate);
        Assert.assertEquals("Learning rate setter should update the learning rate",
                            newLearningRate, optimizer.getLearningRate(), 0.0);
    }
}
