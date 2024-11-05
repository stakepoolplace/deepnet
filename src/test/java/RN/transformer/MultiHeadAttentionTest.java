package RN.transformer;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;

public class MultiHeadAttentionTest {

    @Test
    public void backward0Test() {
        int dModel = 300;
        int numHeads = 6;
        int depth = dModel / numHeads;
        int seqLength = 1;
        int batchSize = 1;

        // Création d'une instance de MultiHeadAttention
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Entrées fictives
        // Création d'une entrée fictive [batchSize=1, seqLength=1, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);


        // Masque fictif (optionnel)
        INDArray mask = null; // Ou créez un masque de forme [numHeads, seqLength, seqLength] si nécessaire

        // Passe forward
        INDArray output = mha.forward(input, input, input, mask);

        
        // Vérification des dimensions de la sortie
        assertEquals(3, output.rank(), "Le tenseur devrait être d'ordre 3");

        // Création d'un gradOutput fictif
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, 1, 1, dModel);

        // Passe backward
        Map<String, INDArray> gradients = mha.backward(gradOutput);

        // Assertions pour vérifier les gradients
        assertNotNull(gradients, "Les gradients ne devraient pas être null");
        assertFalse(gradients.isEmpty(), "Le map des gradients ne devrait pas être vide");
        assertTrue(gradients.containsKey("Wq"), "Les gradients devraient contenir la clé 'Wq'");
        assertTrue(gradients.containsKey("Wk"), "Les gradients devraient contenir la clé 'Wk'");
        assertTrue(gradients.containsKey("Wv"), "Les gradients devraient contenir la clé 'Wv'");
        assertTrue(gradients.containsKey("Wo"), "Les gradients devraient contenir la clé 'Wo'");
        assertTrue(gradients.containsKey("input"), "Les gradients devraient contenir la clé 'input'");

        // Vérifier les formes des gradients
        INDArray gradWq = gradients.get("Wq");
        INDArray gradWk = gradients.get("Wk");
        INDArray gradWv = gradients.get("Wv");
        INDArray gradWo = gradients.get("Wo");
        INDArray gradInput = gradients.get("input");

        assertEquals(dModel, gradWq.rows(), "Le nombre de lignes de gradWq devrait correspondre à dModel");
        assertEquals(numHeads * depth, gradWq.columns(), "Le nombre de colonnes de gradWq devrait correspondre à numHeads * depth");

        assertEquals(dModel, gradWk.rows(), "Le nombre de lignes de gradWk devrait correspondre à dModel");
        assertEquals(numHeads * depth, gradWk.columns(), "Le nombre de colonnes de gradWk devrait correspondre à numHeads * depth");

        assertEquals(numHeads * depth, gradWv.rows(), "Le nombre de lignes de gradWv devrait correspondre à numHeads * depth");
        assertEquals(dModel, gradWv.columns(), "Le nombre de colonnes de gradWv devrait correspondre à dModel");

        assertEquals(numHeads * depth, gradWo.rows(), "Le nombre de lignes de gradWo devrait correspondre à numHeads * depth");
        assertEquals(dModel, gradWo.columns(), "Le nombre de colonnes de gradWo devrait correspondre à dModel");

        //assertEquals(seqLength, gradInput.rows(), "Le nombre de lignes de gradInput devrait correspondre à seqLength");
        //assertEquals(dModel, gradInput.columns(), "Le nombre de colonnes de gradInput devrait correspondre à dModel");

        // Optionnel : Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        assertFalse(gradWq.isNaN().any(), "gradWq ne devrait pas contenir de NaN");
        assertFalse(gradWq.isInfinite().any(), "gradWq ne devrait pas contenir d'Inf");
        assertFalse(gradWk.isNaN().any(), "gradWk ne devrait pas contenir de NaN");
        assertFalse(gradWk.isInfinite().any(), "gradWk ne devrait pas contenir d'Inf");
        assertFalse(gradWv.isNaN().any(), "gradWv ne devrait pas contenir de NaN");
        assertFalse(gradWv.isInfinite().any(), "gradWv ne devrait pas contenir d'Inf");
        assertFalse(gradWo.isNaN().any(), "gradWo ne devrait pas contenir de NaN");
        assertFalse(gradWo.isInfinite().any(), "gradWo ne devrait pas contenir d'Inf");
        assertFalse(gradInput.isNaN().any(), "gradInput ne devrait pas contenir de NaN");
        assertFalse(gradInput.isInfinite().any(), "gradInput ne devrait pas contenir d'Inf");
    }


    @Test
    public void forwardTest(){
        int d_model = 300;
        int num_heads = 6;
        MultiHeadAttention mha = new MultiHeadAttention(d_model, num_heads);

        // Création d'une entrée fictive [batchSize=1, seqLength=1, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, d_model);
        System.out.println("Input shape: " + input.shapeInfoToString());

        // Forward pass
        INDArray output = mha.forward(input, input, input, null);
        System.out.println("Output shape: " + output.shapeInfoToString()); // Devrait être [1, 1, 300]

        // Simuler un gradient de sortie
        INDArray gradOutput = Nd4j.ones(DataType.FLOAT, 1, 1, d_model);

        // Backward pass
        Map<String, INDArray> gradients = mha.backward(gradOutput);
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            System.out.println("Gradient for " + entry.getKey() + ": " + entry.getValue().shapeInfoToString());
        }
    }

    @Test
    public void backwardTest(){
        int d_model = 300;
        int num_heads = 6;
        MultiHeadAttention mha = new MultiHeadAttention(d_model, num_heads);

        // Create a dummy input [batchSize=1, seqLength=2, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 2, d_model);
        System.out.println("Input shape: " + input.shapeInfoToString());

        // Forward pass
        INDArray output = mha.forward(input, input, input, null);
        System.out.println("Output shape: " + output.shapeInfoToString()); // [1, 2, 300]

        // Simulate a gradient of output
        INDArray gradOutput = Nd4j.ones(DataType.FLOAT, 1, 2, d_model);
        System.out.println("Grad Output shape: " + gradOutput.shapeInfoToString());

        // Backward pass
        Map<String, INDArray> gradients = mha.backward(gradOutput);
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            System.out.println("Gradient for " + entry.getKey() + ": " + entry.getValue().shapeInfoToString());
        }
    }
    
}
