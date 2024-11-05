package RN.transformer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.Assert.*;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;

public class LayerNormTest {

    @Test
    public void testLayerNormBackward() {
        int seqLength = 2;
        int dModel = 3;

        // Création d'un LayerNorm
        LayerNorm layerNorm = new LayerNorm(dModel);

        // Entrée fictive
        INDArray input = Nd4j.create(new float[][]{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        });

        // Passe forward
        INDArray output = layerNorm.forward(input);

        // Création d'un gradOutput fictif
        INDArray gradOutput = Nd4j.create(new float[][]{
            {0.1f, 0.2f, 0.3f},
            {0.4f, 0.5f, 0.6f}
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
        assertEquals(seqLength, gradInput.rows(), "Le nombre de lignes de gradInput devrait être égal à seqLength");
        assertEquals(dModel, gradInput.columns(), "Le nombre de colonnes de gradInput devrait être égal à dModel");

        // Optionnel : Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        assertFalse(gradGamma.isNaN().any(), "gradGamma ne devrait pas contenir de NaN");
        assertFalse(gradGamma.isInfinite().any(), "gradGamma ne devrait pas contenir d'Inf");
        assertFalse(gradBeta.isNaN().any(), "gradBeta ne devrait pas contenir de NaN");
        assertFalse(gradBeta.isInfinite().any(), "gradBeta ne devrait pas contenir d'Inf");
        assertFalse(gradInput.isNaN().any(), "gradInput ne devrait pas contenir de NaN");
        assertFalse(gradInput.isInfinite().any(), "gradInput ne devrait pas contenir d'Inf");
    }
}
