package RN.transformer;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class LayerNormGradientCheckTest {

    private static final double EPSILON = 1e-5;
    private static final double THRESHOLD = 1e-4;

    @Test
    public void testLayerNormGradient() {
        // Configuration simplifiée
        int dModel = 4; // Dimension du modèle
        LayerNorm layerNorm = new LayerNorm(dModel);

        // Input simple
        INDArray input = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{1, dModel});
        layerNorm.forward(input);

        // Gradient de sortie simulé
        INDArray gradOutput = Nd4j.create(new double[]{0.1, 0.2, 0.3, 0.4}, new int[]{1, dModel});
        layerNorm.backward(gradOutput);

        // Obtenir les gradients analytiques
        INDArray gradGamma = layerNorm.getGradGamma();
        INDArray gradBeta = layerNorm.getGradBeta();

        // Gradient numérique pour Gamma
        for (int i = 0; i < dModel; i++) {
            // Perturbation positive
            INDArray inputPlus = input.dup();
            inputPlus.putScalar(0, i, input.getDouble(0, i) + EPSILON);
            layerNorm.forward(inputPlus);
            double lossPlus = layerNorm.computeLoss();

            // Perturbation négative
            INDArray inputMinus = input.dup();
            inputMinus.putScalar(0, i, input.getDouble(0, i) - EPSILON);
            layerNorm.forward(inputMinus);
            double lossMinus = layerNorm.computeLoss();

            // Gradient numérique
            double numericGrad = (lossPlus - lossMinus) / (2 * EPSILON);

            // Gradient analytique
            double analyticGrad = gradGamma.getDouble(i);

            // Comparaison
            double diff = Math.abs(numericGrad - analyticGrad);
            System.out.printf("Gamma[%d] - Numeric: %.6f, Analytic: %.6f, Diff: %.6f%n",
                    i, numericGrad, analyticGrad, diff);
            assertTrue("Gradient Gamma mismatch at index " + i, diff < THRESHOLD);
        }

        // Répétez le processus pour Beta si nécessaire
        // (similaire à Gamma)
    }
}
