package RN.transformer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class DropoutTest {

    private static final double THRESHOLD = 0.1;

    @Test
    public void testApplyTrainingWithFixedMask() {
        double rate = 0.2;
        Dropout dropout = new Dropout(rate);
        INDArray input = Nd4j.ones(2, 2); // [2, 2] remplis de 1.0

        // Définir un masque fixe
        INDArray fixedMask = Nd4j.create(new double[][] {
            {1.0, 1.0},
            {0.0, 1.0}
        });

        INDArray output = dropout.apply(true, input, fixedMask);

        // Vérifier que le masque est appliqué
        assertNotNull(dropout.mask, "Le masque ne doit pas être null en mode entraînement.");

        // Afficher le masque et la sortie
        System.out.println("Mask:\n" + dropout.mask);
        System.out.println("Output:\n" + output);

        // Calcul attendu avec Inverted Dropout
        // output = input * mask / (1 - rate) = [[1/0.8, 1/0.8], [0, 1/0.8]] = [[1.25, 1.25], [0.0, 1.25]]
        double mean = output.meanNumber().doubleValue();
        System.out.printf("Mean of activations: %.4f%n", mean);
        assertEquals(1.0, mean, THRESHOLD, "La moyenne des activations devrait être proche de 1.0 avec Inverted Dropout.");
    }

    @Test
    public void testApplyTrainingRandomMask() {
        double rate = 0.2;
        Dropout dropout = new Dropout(rate);
        INDArray input = Nd4j.ones(1000, 1000); // [1000, 1000] remplis de 1.0

        // Fixer la graîne pour la reproductibilité
        Nd4j.getRandom().setSeed(12345);

        INDArray output = dropout.apply(true, input);

        // Vérifier que le masque est appliqué
        assertNotNull(dropout.mask, "Le masque ne doit pas être null en mode entraînement.");

        // Calcul attendu avec Inverted Dropout
        // La moyenne des activations devrait être proche de 1.0
        double mean = output.meanNumber().doubleValue();
        System.out.printf("Mean of activations: %.4f%n", mean);
        assertEquals(1.0, mean, 0.05, "La moyenne des activations devrait être proche de 1.0 avec Inverted Dropout.");
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
        // Fixer la graîne pour la reproductibilité
        Nd4j.getRandom().setSeed(12345);
        INDArray mask = Nd4j.create(new double[][] {
            {1.0, 1.0},
            {0.0, 1.0}
        });

        INDArray output = dropout.apply(true, input, mask);

        // Simuler des gradients de sortie
        INDArray gradOutput = Nd4j.create(new double[][] {
            {1.0, 1.0},
            {1.0, 1.0}
        });

        INDArray gradInput = dropout.backward(gradOutput);

        // Calculer le masque attendu
        // Avec graîne 12345 et taux 0.2, le masque pourrait être :
        // [[1.0, 1.0],
        //  [0.0, 1.0]]
        // Ainsi, gradInput = mask * gradOutput / (1 - rate)
        INDArray expectedGradInput = Nd4j.create(new double[][] {
            {1.25, 1.25},
            {0.0, 1.25}
        });

        System.out.println("expectedGradInput: " + expectedGradInput);
        System.out.println("gradInput: " + gradInput);

        // Vérifier les gradients
        assertTrue(expectedGradInput.equalsWithEps(gradInput, THRESHOLD), 
            "Les gradients d'entrée doivent correspondre aux gradients attendus.");
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