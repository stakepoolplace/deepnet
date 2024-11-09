package RN.transformer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;
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
    


}
