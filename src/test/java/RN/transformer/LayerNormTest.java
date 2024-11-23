package RN.transformer;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;

public class LayerNormTest {

    @Test
    public void testLayerNormBackward() {
        int batchSize = 1;
        int seqLength = 2;
        int dModel = 3;
    
        // Création d'un LayerNorm
        LayerNorm layerNorm = new LayerNorm(dModel);
    
        // Entrée fictive avec trois dimensions [batchSize, seqLength, dModel]
        INDArray input = Nd4j.create(new float[][][]{
            {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}
            }
        });
    
        // Passe forward
        INDArray output = layerNorm.forward(input);
    
        // Création d'un gradOutput fictif avec trois dimensions [batchSize, seqLength, dModel]
        INDArray gradOutput = Nd4j.create(new float[][][]{
            {
                {0.1f, 0.2f, 0.3f},
                {0.4f, 0.5f, 0.6f}
            }
        });
    
        // Passe backward
        Map<String, INDArray> gradients = layerNorm.backward(gradOutput);
    
        // Assertions
        assertNotNull(gradients, "Les gradients ne devraient pas être null");
        assertFalse(gradients.isEmpty(), "Le map des gradients ne devrait pas être vide");
        assertTrue(gradients.containsKey("gamma"), "Les gradients devraient contenir la clé 'gamma'");
        assertTrue(gradients.containsKey("beta"), "Les gradients devraient contenir la clé 'beta'");
        assertTrue(gradients.containsKey("input"), "Les gradients devraient contenir la clé 'input'");
    
        // Vérifier les formes des gradients
        INDArray gradGamma = gradients.get("gamma");
        INDArray gradBeta = gradients.get("beta");
        INDArray gradInput = gradients.get("input");
    
        assertEquals(dModel, gradGamma.length(), "La longueur de gradGamma devrait être égale à dModel");
        assertEquals(dModel, gradBeta.length(), "La longueur de gradBeta devrait être égale à dModel");
        assertEquals(batchSize, gradInput.size(0), "Le batchSize de gradInput devrait correspondre");
        assertEquals(seqLength, gradInput.size(1), "Le nombre de séquences de gradInput devrait correspondre");
        assertEquals(dModel, gradInput.size(2), "Le nombre de dimensions de gradInput devrait correspondre");
    
        // Optionnel : Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        assertFalse(gradGamma.isNaN().any(), "gradGamma ne devrait pas contenir de NaN");
        assertFalse(gradGamma.isInfinite().any(), "gradGamma ne devrait pas contenir d'Inf");
        assertFalse(gradBeta.isNaN().any(), "gradBeta ne devrait pas contenir de NaN");
        assertFalse(gradBeta.isInfinite().any(), "gradBeta ne devrait pas contenir d'Inf");
        assertFalse(gradInput.isNaN().any(), "gradInput ne devrait pas contenir de NaN");
        assertFalse(gradInput.isInfinite().any(), "gradInput ne devrait pas contenir d'Inf");
    }

    @Test
    public void testLayerNormForwardBackward() {
        int batchSize = 2;
        int seqLength = 3;
        int dModel = 4;

        // Créer une instance de LayerNorm
        LayerNorm layerNorm = new LayerNorm(dModel);

        // Définir une entrée spécifique
        INDArray input = Nd4j.create(new float[][][] {
            {
                {1.0f, 2.0f, 3.0f, 4.0f},
                {2.0f, 3.0f, 4.0f, 5.0f},
                {3.0f, 4.0f, 5.0f, 6.0f}
            },
            {
                {4.0f, 5.0f, 6.0f, 7.0f},
                {5.0f, 6.0f, 7.0f, 8.0f},
                {6.0f, 7.0f, 8.0f, 9.0f}
            }
        }); // [2, 3, 4]

        // Passe forward
        INDArray output = layerNorm.forward(input);

        // Vérifier que la sortie a la même forme que l'entrée
        assertArrayEquals(input.shape(), output.shape(), "La forme de la sortie doit correspondre à celle de l'entrée.");

        // Passe backward avec un gradient de sortie spécifique (par exemple, des uns)
        INDArray gradOutput = Nd4j.ones(DataType.FLOAT, input.shape()); // [2, 3, 4]
        Map<String, INDArray> gradients = layerNorm.backward(gradOutput);

        // Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            String key = entry.getKey();
            INDArray grad = entry.getValue();
            assertFalse(grad.isNaN().any(), "Le gradient " + key + " contient des NaN.");
            assertFalse(grad.isInfinite().any(), "Le gradient " + key + " contient des Inf.");
        }

        // Afficher les gradients pour vérification
        System.out.println("gradGamma:\n" + gradients.get("gamma"));
        System.out.println("gradBeta:\n" + gradients.get("beta"));
        System.out.println("gradInput:\n" + gradients.get("input"));

        // Assertions supplémentaires selon les calculs attendus
        // ...
    }
    


}
