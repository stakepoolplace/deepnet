package RN.transformer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GradientClippingTest {


    @Test
    public void testGradientClipping() {
        // Création de gradients avec certaines normes élevées
        INDArray grad1 = Nd4j.create(new double[]{3.0, 4.0}); // Norme 5.0
        INDArray grad2 = Nd4j.create(new double[]{6.0, 8.0}); // Norme 10.0
        List<INDArray> gradients = Arrays.asList(grad1, grad2);

        // Seuil de clipping
        double maxNorm = 5.0;

        // Fonction de clipping
        GradientClipper clipper = new GradientClipper();
        clipper.clipGradients(gradients, maxNorm);

        // Vérification des normes après clipping
        double norm1 = grad1.norm2Number().doubleValue();
        double norm2 = grad2.norm2Number().doubleValue();

        System.out.printf("Norme Gradient1 après clipping: %.6f%n", norm1);
        System.out.printf("Norme Gradient2 après clipping: %.6f%n", norm2);

        // Les normes devraient être <= maxNorm
        assertTrue(norm1 <= maxNorm + 1e-6);
        assertTrue(norm2 <= maxNorm + 1e-6);
    }

    
}
