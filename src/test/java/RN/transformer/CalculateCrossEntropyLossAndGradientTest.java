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
        int dModel = 300; // vocabSize = 3 pour simplifier
        TransformerModel model = new TransformerModel(1, dModel, 1, 127, 0, 0.001f, 10);
    
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
    
        // targetBatch : [1, 2] IDs des tokens cibles (0 est <PAD>, 1 est <UNK>)
        INDArray targetBatch = Nd4j.create(new double[][] { {0, 1} }); // Premier token : <PAD>, deuxième token : <UNK>
    
        // Vérifier les dimensions
        assertEquals(1, logits.size(0), "Batch size incorrect");
        assertEquals(2, logits.size(1), "Seq length incorrect");
        assertEquals(3, logits.size(2), "Vocab size incorrect");
    
        assertEquals(1, targetBatch.size(0), "Batch size incorrect pour targetBatch");
        assertEquals(2, targetBatch.size(1), "Seq length incorrect pour targetBatch");
    
        // Afficher les données d'entrée
        System.out.println("Logits: " + logits);
        System.out.println("Target Batch: " + targetBatch);
    
        // Appel de la méthode à tester
        Pair<Float, INDArray> result = model.calculateCrossEntropyLossAndGradient(decodedLogits, targetBatch);
        float loss = result.getLeft();
        INDArray gradients = result.getRight();
    
        // Probabilités calculées (pour référence)
        INDArray probabilities = Nd4j.create(new double[][][] {
            { {0.6590, 0.2424, 0.0986},
              {0.2537, 0.6897, 0.0566} }
        }); // [1, 2, 3]
    
        // Calcul manuel de la perte (seulement le deuxième token)
        double expectedLoss = -Math.log(probabilities.getDouble(0, 1, 1)); // ≈0.3715
        float expectedLossFloat = (float) expectedLoss;
    
        System.out.printf("Computed Loss: %.6f, Expected Loss: %.6f%n", loss, expectedLossFloat);
        assertEquals(expectedLossFloat, loss, THRESHOLD, "La perte calculée ne correspond pas à la perte attendue.");
    
        // Calcul manuel des gradients
        INDArray expectedGradients = Nd4j.create(new double[][][] {
            { {0.0, 0.0, 0.0},
              {0.2537, -0.3103, 0.0566} }
        }); // [1,2,3]
    
        System.out.println("Computed Gradients: " + gradients);
        System.out.println("Expected Gradients: " + expectedGradients);
    
        // Vérifier que les gradients sont corrects
        // Utiliser assertTrue avec isClose ou equalsWithEps pour comparer les INDArray
        assertTrue(expectedGradients.equalsWithEps(gradients, THRESHOLD), 
        "Les gradients calculés ne correspondent pas aux gradients attendus.");
            }
}
