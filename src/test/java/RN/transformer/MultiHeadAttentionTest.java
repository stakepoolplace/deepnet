package RN.transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;
import java.util.List;
import java.util.Arrays;
import org.apache.commons.lang3.tuple.Pair;

public class MultiHeadAttentionTest {

    private static TransformerModel transformerModel;
    private static int dModel;
    private static int numHeads;
    private static int vocabSize;

    @BeforeAll
    public static void setup() {
        // Initialisation des paramètres
        dModel = 300;
        numHeads = 6;
        vocabSize = 10000; // Exemple de taille de vocabulaire

    }

    @Test
    public void testSoftmaxGrad() {

        int depth = dModel / numHeads;
        int seqLength = 1;
        int batchSize = 1;

        // Création d'une instance de MultiHeadAttention
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Créer un softmaxOutput connu
        INDArray softmaxOutput = Nd4j.create(new double[][]{
            {0.3, 0.7},
            {0.6, 0.4}
        }).reshape(1, 1, 2, 2); // [batchSize=1, numHeads=1, seqLength=2, seqLength=2]
    
        // Créer un gradOutput connu
        INDArray gradOutput = Nd4j.create(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        }).reshape(1, 1, 2, 2); // [batchSize=1, numHeads=1, seqLength=2, seqLength=2]
    
        // Calculer gradScores
        INDArray gradScores = mha.softmaxGrad(softmaxOutput, gradOutput);
    
        // Calculer gradScores manuellement pour comparaison
        // gradScores = S * dL/dS - S * sum(S * dL/dS, axis=-1)
        INDArray expectedSum = Nd4j.create(new double[][]{
            {0.3 * 1.0 + 0.7 * 2.0}, // For first row
            {0.6 * 3.0 + 0.4 * 4.0}  // For second row
        }).reshape(1, 1, 2, 1); // [1,1,2,1]
    
        INDArray expectedGradScores = softmaxOutput.mul(gradOutput).sub(softmaxOutput.mul(expectedSum));
    
        // Assert que gradScores est égal à expectedGradScores avec une tolérance
        // Assert que gradScores est égal à expectedGradScores avec une tolérance
        assertTrue("gradScores should match the expected gradients.",
                   expectedGradScores.equalsWithEps(gradScores, 1e-6));    }
    

    @Test
    public void backward0Test() {
        int depth = dModel / numHeads;
        int seqLength = 1;
        int batchSize = 1;

        // Création d'une instance de MultiHeadAttention
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Entrées fictives
        // Création d'une entrée fictive [batchSize=1, seqLength=1, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);

        // Masque fictif (optionnel)
        INDArray mask = null; // Ou créez un masque de forme [batchSize, 1, 1, seqLength] si nécessaire

        // Passe forward
        INDArray output = mha.forward(input, input, input, mask);

        // Vérification des dimensions de la sortie
        assertEquals(3, output.rank(), "Le tenseur devrait être d'ordre 3");
        assertArrayEquals(new long[]{batchSize, seqLength, dModel}, output.shape(), "La forme de la sortie est incorrecte");

        // Création d'un gradOutput fictif
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);

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

        // Vérifier les gradients des entrées
        assertEquals(batchSize, gradInput.shape()[0], "Le nombre de batch de gradInput est incorrect");
        assertEquals(seqLength, gradInput.shape()[1], "La longueur de séquence de gradInput est incorrecte");
        assertEquals(dModel, gradInput.shape()[2], "La dimension du modèle de gradInput est incorrecte");

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
        int depth = dModel / numHeads;
        int seqLength = 1;
        int batchSize = 1;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Création d'une entrée fictive [batchSize=1, seqLength=1, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);
        System.out.println("Input shape: " + Arrays.toString(input.shape()));

        // Forward pass
        INDArray output = mha.forward(input, input, input, null);
        System.out.println("Output shape: " + Arrays.toString(output.shape())); // Devrait être [1, 1, 300]

        // Vérifier les dimensions de la sortie
        assertEquals(3, output.rank(), "Le tenseur devrait être d'ordre 3");
        assertArrayEquals(new long[]{batchSize, seqLength, dModel}, output.shape(), "La forme de la sortie est incorrecte");
    }

    @Test
    public void testBackward() {
        // Initialiser les paramètres avec des valeurs aléatoires
        int batchSize = 1;
        int seqLength = 6;
        int numHeads = 2;
        int depth = 150;
        int dModel = numHeads * depth;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        mha.initializeWeights(); // Méthode pour initialiser Wq, Wk, Wv, Wo


        // Créer des entrées simples avec forme [batchSize, seqLength, dModel]
        INDArray inputQ = Nd4j.randn(batchSize, seqLength, dModel);
        INDArray inputK = Nd4j.randn(batchSize, seqLength, dModel);
        INDArray inputV = Nd4j.randn(batchSize, seqLength, dModel);

        // Effectuer la passe forward
        INDArray output = mha.forward(inputQ, inputK, inputV, null);

        // Créer un gradient de sortie simple
        INDArray gradOutput = Nd4j.randn(batchSize, seqLength, dModel);

        // Effectuer la passe backward
        Map<String, INDArray> gradients = mha.backward(gradOutput);

        // Vérifier que les gradients pour Wq, Wk, Wv, Wo ne sont pas nuls
        assertNotNull("Gradient Wq ne doit pas être null.", gradients.get("Wq"));
        assertNotNull("Gradient Wk ne doit pas être null.", gradients.get("Wk"));
        assertNotNull("Gradient Wv ne doit pas être null.", gradients.get("Wv"));
        assertNotNull("Gradient Wo ne doit pas être null.", gradients.get("Wo"));

        // Vérifier que les gradients ne sont pas tous à zéro
        assertTrue("Gradient Wq ne doit pas être zéro.", gradients.get("Wq").sumNumber().doubleValue() != 0.0);
        assertTrue("Gradient Wk ne doit pas être zéro.", gradients.get("Wk").sumNumber().doubleValue() != 0.0);
        assertTrue("Gradient Wv ne doit pas être zéro.", gradients.get("Wv").sumNumber().doubleValue() != 0.0);
        assertTrue("Gradient Wo ne doit pas être zéro.", gradients.get("Wo").sumNumber().doubleValue() != 0.0);
    }

    @Test
    public void backwardTest(){
        int depth = dModel / numHeads;
        int seqLength = 2;
        int batchSize = 1;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Create a dummy input [batchSize=1, seqLength=2, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);
        System.out.println("Input shape: " + Arrays.toString(input.shape()));

        // Forward pass
        INDArray output = mha.forward(input, input, input, null);
        System.out.println("Output shape: " + Arrays.toString(output.shape())); // [1, 2, 300]

        // Simulate a gradient of output
        INDArray gradOutput = Nd4j.ones(DataType.FLOAT, batchSize, seqLength, dModel);
        System.out.println("Grad Output shape: " + Arrays.toString(gradOutput.shape()));

        // Backward pass
        Map<String, INDArray> gradients = mha.backward(gradOutput);
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            System.out.println("Gradient for " + entry.getKey() + ": " + Arrays.toString(entry.getValue().shape()));
        }

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

        // Vérifier les gradients des entrées
        assertEquals(batchSize, gradInput.shape()[0], "Le nombre de batch de gradInput est incorrect");
        assertEquals(seqLength, gradInput.shape()[1], "La longueur de séquence de gradInput est incorrecte");
        assertEquals(dModel, gradInput.shape()[2], "La dimension du modèle de gradInput est incorrecte");

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
    public void multiBatchTest(){
        int depth = dModel / numHeads;
        int seqLength = 3;
        int batchSize = 2;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Création d'une entrée fictive [batchSize=2, seqLength=3, dModel=300]
        INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);
        System.out.println("Input shape: " + Arrays.toString(input.shape()));

        // Forward pass
        INDArray output = mha.forward(input, input, input, null);
        System.out.println("Output shape: " + Arrays.toString(output.shape())); // Devrait être [2, 3, 300]

        // Vérifier les dimensions de la sortie
        assertEquals(3, output.rank(), "Le tenseur devrait être d'ordre 3");
        assertArrayEquals(new long[]{batchSize, seqLength, dModel}, output.shape(), "La forme de la sortie est incorrecte");

        // Création d'un gradOutput fictif [batchSize=2, seqLength=3, dModel=300]
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);

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

        // Vérifier les gradients des entrées
        assertEquals(batchSize, gradInput.shape()[0], "Le nombre de batch de gradInput est incorrect");
        assertEquals(seqLength, gradInput.shape()[1], "La longueur de séquence de gradInput est incorrecte");
        assertEquals(dModel, gradInput.shape()[2], "La dimension du modèle de gradInput est incorrecte");

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
}
