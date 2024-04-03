package RN.transformer;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoAnnotations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdamUpdater;

public class CustomAdamOptimizerTest {

    private CustomAdamOptimizer optimizer;
    private List<INDArray> parameters;
    private INDArray gradients;
    private final double initialLr = 0.001;
    private final int warmupSteps = 1000;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        
        // Mock AdamUpdater to avoid initialization issue
        AdamUpdater mockedAdamUpdater = mock(AdamUpdater.class);
        // Assume update method would modify parameters in a certain way, define that behavior here
        // For example, do nothing (this is a simplification, you should define behavior close to reality)
        doNothing().when(mockedAdamUpdater).applyUpdater(any(), anyInt(), anyInt());
        
        optimizer = new CustomAdamOptimizer(initialLr, warmupSteps); // Adjust constructor as needed
        
        // Initialize parameters and gradient with dummy values
        parameters = new ArrayList<>();
        parameters.add(Nd4j.create(new float[]{0.1f, -0.2f}, new int[]{1, 2}));
        gradients = Nd4j.create(new float[]{0.01f, -0.01f}, new int[]{1, 2});
    }

    @Test
    public void testUpdate() {
        INDArray originalParams = parameters.get(0).dup();
        optimizer.update(parameters, gradients);
        INDArray updatedParams = parameters.get(0);

        // Assert that parameters have been updated (changed)
        Assert.assertNotEquals("Parameters should be updated after optimizer update call",
                               originalParams, updatedParams);
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
                          learningRateAfterWarmup < learningRateMidway);
    }
    
    @Test
    public void testLearningRateSetter() {
        double newLearningRate = 0.0001;
        optimizer.setLearningRate(newLearningRate);
        Assert.assertEquals("Learning rate setter should update the learning rate",
                            newLearningRate, optimizer.getLearningRate(), 0.0);
    }
}
