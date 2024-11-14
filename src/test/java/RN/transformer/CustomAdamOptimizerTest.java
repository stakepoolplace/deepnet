package RN.transformer;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CustomAdamOptimizerTest {

    private CustomAdamOptimizer optimizer;
    private List<INDArray> parameters;
    private List<INDArray> gradients;
    private final float initialLr = 0.001f;
    private final int warmupSteps = 1000;
    private final int dModel = 512;
    

    @Before
    public void setUp() {

        
        // Ensuite, initialiser 'parameters' avec les valeurs de test
        parameters = new ArrayList<>();
        parameters.add(Nd4j.create(new float[]{0.1f, -0.2f}, new int[]{1, 2}));
        gradients = new ArrayList<>();
        gradients.add(Nd4j.create(new float[]{0.01f, -0.01f}, new int[]{1, 2}));
        
        optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, parameters);

    }


    @Test
    public void testAdamUpdate() {
        // Initialisation des paramètres
        INDArray param = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray grad = Nd4j.create(new double[]{0.1, 0.2, 0.3});
        List<INDArray> params = Arrays.asList(param);
        List<INDArray> grads = Arrays.asList(grad);

        // Initialisation de l'optimiseur
        CustomAdamOptimizer optimizer = new CustomAdamOptimizer(0.001f, 16, 10, params);

        // Sauvegarder les valeurs initiales
        INDArray paramBefore = param.dup();

        // Mise à jour
        optimizer.update(params, grads);

        // Vérifier que les paramètres ont changé
        assertFalse(paramBefore.equals(param));

        // Afficher les changements
        System.out.println("Param before update: " + paramBefore);
        System.out.println("Param after update: " + param);
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

        float learningRateAfterWarmup = optimizer.getLearningRate();

        // Verify learning rate after warmup is less than during warmup
        Assert.assertTrue("Learning rate should decrease after warmup",
                          learningRateAfterWarmup < initialLr);
    }
    
    @Test
    public void testLearningRateSetter() {
        float newLearningRate = 0.0001f;
        optimizer.setLearningRate(newLearningRate);
        Assert.assertEquals("Learning rate setter should update the learning rate",
                            newLearningRate, optimizer.getLearningRate(), 0.0);
    }
}
