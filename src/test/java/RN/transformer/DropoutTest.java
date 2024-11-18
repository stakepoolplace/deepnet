package RN.transformer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class DropoutTest {

    @Test
    public void testApplyTraining() {
        double rate = 0.2;
        Dropout dropout = new Dropout(rate);
        INDArray input = Nd4j.ones(2, 2); // [2, 2] remplis de 1.0

        INDArray output = dropout.apply(true, input);

        // Vérifier que le masque est appliqué
        assertNotNull(dropout.mask, "Le masque ne doit pas être null en mode entraînement.");

        // Calcul attendu avec Inverted Dropout
        // output = input * mask / (1 - rate)
        // La moyenne des activations devrait être proche de 1.0
        double mean = output.meanNumber().doubleValue();
        assertEquals(1.0, mean, 0.1, "La moyenne des activations devrait être proche de 1.0 avec Inverted Dropout.");
    }

    @Test
    public void testApplyInference() {
        double rate = 0.2;
        Dropout dropout = new Dropout(rate);
        INDArray input = Nd4j.create(new double[][] {
            {1.0, 2.0},
            {3.0, 4.0}
        }); // [2, 2]

        INDArray output = dropout.apply(false, input);

        // En inférence, le dropout n'est pas appliqué
        assertEquals(input, output, "En inférence, le dropout ne doit pas être appliqué.");
    }

    @Test
    public void testBackward() {
        double rate = 0.2;
        Dropout dropout = new Dropout(rate);
        INDArray input = Nd4j.ones(2, 2); // [2, 2]
        INDArray output = dropout.apply(true, input);

        // Simuler des gradients de sortie
        INDArray gradOutput = Nd4j.create(new double[][] {
            {1.0, 1.0},
            {1.0, 1.0}
        });

        INDArray gradInput = dropout.backward(gradOutput);

        // gradInput devrait être gradOutput * mask / (1 - rate)
        INDArray expectedGradInput = gradOutput.mul(dropout.mask).div(1.0 - rate);
        assertEquals(expectedGradInput, gradInput, "Les gradients d'entrée doivent correspondre aux gradients attendus.");
    }

    @Test
    public void testDropoutRateValidation() {
        // Taux de dropout invalide (négatif)
        assertThrows(IllegalArgumentException.class, () -> {
            new Dropout(-0.1);
        }, "Le taux de dropout négatif devrait lancer une exception.");

        // Taux de dropout invalide (>= 1)
        assertThrows(IllegalArgumentException.class, () -> {
            new Dropout(1.0);
        }, "Le taux de dropout >= 1 devrait lancer une exception.");
    }
}
