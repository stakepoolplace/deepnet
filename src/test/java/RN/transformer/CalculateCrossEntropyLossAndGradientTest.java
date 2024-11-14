package RN.transformer;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

public class CalculateCrossEntropyLossAndGradientTest {

    private static final double THRESHOLD = 1e-4;

    @Test
    public void testCalculateCrossEntropyLossAndGradient() {
        // Configuration du modèle
        int dModel = 3; // vocabSize = 3 pour simplifier
        TransformerModel model = new TransformerModel(dModel);

        // Initialisation de l'optimiseur
        CustomAdamOptimizer optimizer = new CustomAdamOptimizer(0.001f, dModel, 10, model.getParameters());
        model.setOptimizer(optimizer);

        // Entrées connues
        // decodedLogits : [1, 2, 3] batchSize=1, seqLength=2, vocabSize=3
        INDArray logits = Nd4j.create(new double[][][] {
            { {2.0, 1.0, 0.1}, 
              {0.5, 1.5, -1.0} }
        }); // Shape: [1, 2, 3]
        List<INDArray> decodedLogits = Arrays.asList(logits);

        // targetBatch : [1, 2] IDs des tokens cibles
        INDArray targetBatch = Nd4j.create(new double[][] { {0, 1} }); // Premier token : classe 0, deuxième token : classe 1

        // Appel de la méthode à tester
        Pair<Float, INDArray> result = model.calculateCrossEntropyLossAndGradient(decodedLogits, targetBatch);
        float loss = result.getLeft();
        INDArray gradients = result.getRight();

        // Calcul manuel des probabilités
        INDArray probabilities = Nd4j.create(new double[][][] {
            { {0.65900114, 0.24243297, 0.09856589},
              {0.37754067, 0.5761169 , 0.04634243} }
        }); // Calculé avec softmax

        // Calcul manuel de la perte
        double expectedLoss = -Math.log(probabilities.getDouble(0, 0, 0)) - Math.log(probabilities.getDouble(0, 1, 1));
        expectedLoss /= 2.0; // Moyenne sur batchSize=1, seqLength=2

        System.out.printf("Computed Loss: %.6f, Expected Loss: %.6f%n", loss, expectedLoss);
        assertEquals(expectedLoss, loss, THRESHOLD, "La perte calculée ne correspond pas à la perte attendue.");

        // Calcul manuel des gradients
        INDArray targetOneHot = Nd4j.create(new double[][][] {
            { {1.0, 0.0, 0.0},
              {0.0, 1.0, 0.0} }
        }); // One-hot encoding

        INDArray expectedGradients = probabilities.sub(targetOneHot).div(2.0); // [1, 2, 3]

        System.out.println("Computed Gradients: " + gradients);
        System.out.println("Expected Gradients: " + expectedGradients);

        // Vérifier que les gradients sont corrects
        assertEquals(expectedGradients, gradients, "Les gradients calculés ne correspondent pas aux gradients attendus.");
    }
}
