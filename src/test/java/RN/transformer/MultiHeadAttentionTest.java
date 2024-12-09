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
import org.nd4j.linalg.indexing.NDArrayIndex;

import RN.utils.NDArrayUtils;

import java.util.Map;
import java.util.Arrays;

public class MultiHeadAttentionTest {

    private static TransformerModel transformerModel;
    private static int dModel;
    private static int numHeads;
    private static int numLayers;
    private static int vocabSize;
    private static int dff;
    private static int maxSequenceLength;
    private static float dropoutRate;
    private static float initialLr;
    private static int warmupSteps;
    private static Tokenizer tokenizer = null;
    private static TransformerModel transformer = null;

    @BeforeAll
    public static void setup() {

        // Initialisation des paramètres
        dModel = 300;
        numHeads = 6;
        numLayers = 6;
        dff = 2048;
        vocabSize = 10000;
        dropoutRate = 0.0f;
        maxSequenceLength = 50;
        initialLr = 0.001f;
        warmupSteps = 1000;
        tokenizer = new Tokenizer(Arrays.asList("<PAD>", "Hello", "world"), dModel, maxSequenceLength);
        
        // Créer une instance du TransformerModel avec le tokenizer
        transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, 3, tokenizer, initialLr, warmupSteps);
        
    }
    

    @Test
    public void testSoftmaxGrad() {

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
        int seqLength = 5;
        int batchSize = 1;

        // Création d'une instance de MultiHeadAttention
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        // Entrées fictives
        // Création d'une entrée fictive [batchSize=1, seqLength=1, dModel=300]
        //INDArray input = Nd4j.rand(DataType.FLOAT, batchSize, seqLength, dModel);
        // Création de tokens fictifs [batchSize=1, seqLength=1]
        // Création de tokens fictifs [batchSize=1, seqLength=5]
        INDArray tokens = Nd4j.createFromArray(new int[][] { { 2, 1, 0, 3, 0 } }); // IDs de tokens

        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokens);

        // Passe forward
        INDArray output = mha.forward(input, input, input, mask, mask, null);

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

        int seqLength = 5;
        int batchSize = 1;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        INDArray tokens = Nd4j.createFromArray(new int[][] { { 2, 1, 0, 3, 0 } }); // IDs de tokens

        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokens);
        
        // Passe forward
        INDArray output = mha.forward(input, input, input, mask, mask, null);
        System.out.println("Output shape: " + Arrays.toString(output.shape())); // Devrait être [1, 1, 300]

        // Vérifier les dimensions de la sortie
        assertEquals(3, output.rank(), "Le tenseur devrait être d'ordre 3");
        assertArrayEquals(new long[]{batchSize, seqLength, dModel}, output.shape(), "La forme de la sortie est incorrecte");
    }

    @Test
    public void testBackward() {
        // Initialiser les paramètres avec des valeurs aléatoires
        int batchSize = 1;
        int seqLength = 5;
        int numHeads = 2;
        int depth = 150;
        int dModel = numHeads * depth;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        mha.initializeWeights(); // Méthode pour initialiser Wq, Wk, Wv, Wo


        // Créer des entrées simples avec forme [batchSize, seqLength, dModel]
        INDArray inputQ = Nd4j.randn(batchSize, seqLength, dModel);
        INDArray inputK = Nd4j.randn(batchSize, seqLength, dModel);
        INDArray inputV = Nd4j.randn(batchSize, seqLength, dModel);

        INDArray tokens = Nd4j.createFromArray(new int[][] { { 2, 1, 0, 3, 0 } }); // IDs de tokens

        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokens);
        

        // Effectuer la passe forward
        INDArray output = mha.forward(inputQ, inputK, inputV, mask, mask, null);

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
        int seqLength = 5;
        int batchSize = 1;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        
        // Create a dummy input [batchSize=1, seqLength=2, dModel=300]
        INDArray tokens = Nd4j.createFromArray(new int[][] { { 2, 1, 0, 3, 0 } }); // IDs de tokens

        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokens);


       // Créer des entrées simples avec forme [batchSize, seqLength, dModel]
       INDArray inputQ = Nd4j.randn(batchSize, seqLength, dModel);
       INDArray inputK = Nd4j.randn(batchSize, seqLength, dModel);
       INDArray inputV = Nd4j.randn(batchSize, seqLength, dModel);


       // Effectuer la passe forward
       INDArray output = mha.forward(inputQ, inputK, inputV, mask, mask, null);

        // Backward pass
        Map<String, INDArray> gradients = mha.backward(output);
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
        int seqLength = 6;
        int batchSize = 2;

        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);

        INDArray tokens = Nd4j.createFromArray(new int[][] { { 2, 1, 0, 3, 0, 0 }, { 2, 1, 0, 3, 0, 0 } }); // IDs de tokens

        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createQueryPaddingMask(tokenizer, tokens);
        
        // Forward pass
        INDArray output = mha.forward(input, input, input, mask, mask, null);
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

        assertEquals(dModel, gradWk.rows(), "Le nombre de lignes de gradWk devrait correspondre  dModel");
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
    public void testCreateLookAheadMask_Binary() {
        
        int batchSize = 2;
        int size = 3;
        INDArray mask = NDArrayUtils.createLookAheadMask(batchSize, size);
        
        // Masque attendu pour size=3
        // [
        //   [
        //     [
        //       [1.0, 0.0, 0.0],
        //       [1.0, 1.0, 0.0],
        //       [1.0, 1.0, 1.0]
        //     ]
        //   ],
        //   [
        //     [
        //       [1.0, 0.0, 0.0],
        //       [1.0, 1.0, 0.0],
        //       [1.0, 1.0, 1.0]
        //     ]
        //   ]
        // ]
        INDArray expectedMask = Nd4j.create(new float[][][][] {
            {
                {
                    {1.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 0.0f},
                    {1.0f, 1.0f, 1.0f}
                }
            },
            {
                {
                    {1.0f, 0.0f, 0.0f},
                    {1.0f, 1.0f, 0.0f},
                    {1.0f, 1.0f, 1.0f}
                }
            }
        }); // [2, 1, 3, 3]
        
        assertTrue(mask.equalsWithEps(expectedMask, 1e-6), "Le masque look-ahead binaire est incorrect.");
    }

    @Test
    public void testCreateLookAheadMask_SingleBatch_SmallSize() {
        int batchSize = 1;
        int size = 3;

        // Générer le masque look-ahead
        INDArray mask = NDArrayUtils.createLookAheadMask(batchSize, size);

        // Définir le masque attendu
        INDArray expectedMask = Nd4j.create(new float[][][][] {
            {
                {
                    {1.0f, 0f, 0f},
                    {1.0f, 1.0f, 0f},
                    {1.0f, 1.0f, 1.0f}
                }
            }
        });

        printMask(expectedMask);
        printMask(mask);

        // Vérifier la forme du masque
        assertArrayEquals(new long[] {batchSize, 1, size, size}, mask.shape(),
        "La forme du masque look-ahead est incorrecte.");

        // Vérifier les valeurs du masque
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    float expectedValue = expectedMask.getFloat(b, 0, i, j);
                    float actualValue = mask.getFloat(b, 0, i, j);
                    assertEquals(expectedValue, actualValue, 1e-6,
                    String.format("Position [%d][%d][%d][%d] devrait être -Infinity, mais était %.2f",
                        b, 0, i, j, actualValue));
                }
            }
        }

        System.out.println("Test réussi : Le masque look-ahead pour un batch unique et petite taille fonctionne correctement.");
    }


    /**
     * Méthode utilitaire pour afficher un masque.
     * Utile pour le debugging mais non utilisée dans les assertions.
     */
    private void printMask(INDArray mask) {
        System.out.println("Mask Shape: " + Arrays.toString(mask.shape()));
        System.out.println(mask);
    }


    @Test
    public void testForwardWithLookAheadMask_SpecificValues() {
        // Configuration des paramètres pour le test
        int dModel = 4;      // Dimension du modèle réduite pour le test
        int numHeads = 2;    // Utiliser 2 têtes au lieu de 1 pour mieux tester le multi-head
        int depth = dModel / numHeads; // Profondeur par tête = 2
        int maxSequenceLength = 4;
        int batchSize = 1;
        
        // Création d'une instance de MultiHeadAttention avec des poids contrôlés
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        
        // Initialisation des poids avec des valeurs spécifiques et contrôlées
        INDArray Wq = Nd4j.create(new float[][] {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        });
        INDArray Wk = Wq.dup();
        INDArray Wv = Wq.dup();
        INDArray Wo = Wq.dup();
        
        // Injection des poids dans l'objet MultiHeadAttention
        mha.setWeights(Wq, Wk, Wv, Wo);
        
        // Création d'une séquence d'entrée simple avec des valeurs connues
        INDArray input = Nd4j.create(new float[][][] {
            {
                {1, 0, 0, 0},  // Premier token
                {0, 1, 0, 0},  // Deuxième token
                {0, 0, 1, 0},  // Troisième token
                {0, 0, 0, 1}   // Quatrième token
            }
        });
        
        // Création du masque look-ahead
        INDArray mask = NDArrayUtils.createLookAheadMask(batchSize, maxSequenceLength);
        
        // Exécution du forward pass
        INDArray output = mha.forward(input, input, input, mask, null, null);
        
        // Vérifications des dimensions de sortie
        assertArrayEquals(
            new long[] {batchSize, maxSequenceLength, dModel},
            output.shape(),
            "La forme de sortie devrait être [batchSize, seqLength, dModel]"
        );
        
        // Vérification des valeurs spécifiques attendues
        // Le premier token devrait avoir la plus forte attention sur lui-même
        assertTrue(output.getFloat(0, 0, 0) > 0.3f, "L'attention du premier token sur lui-même devrait être significative");
        assertTrue(output.getFloat(0, 0, 0) > output.getFloat(0, 0, 1), "Le premier token devrait avoir plus d'attention sur lui-même");
        
        // Le deuxième token devrait voir le premier et lui-même
        assertTrue(output.getFloat(0, 1, 0) > 0, "Le deuxième token devrait voir le premier token");
        assertTrue(output.getFloat(0, 1, 1) > 0, "Le deuxième token devrait se voir lui-même");
        
        // Vérification que l'attention respecte le masquage
        for (int i = 0; i < maxSequenceLength; i++) {
            for (int j = i + 1; j < maxSequenceLength; j++) {
                assertTrue(
                    output.getFloat(0, i, j) < 0.3f,
                    String.format("Position (%d,%d) devrait avoir une attention faible", i, j)
                );
            }
        }
    }


    @Test
    public void testCreatePaddingMask_SpecificValues() {
        
        // Configuration du Tokenizer avec des IDs connus
        // Supposons que <PAD> a l'ID 0

        // Définir un batch avec une séquence de longueur 4 : [1, 2, 0, 0]
        INDArray tokens = Nd4j.createFromArray(new int[][] {
            {1, 2, 0, 0}
        }); // [1, 4]


        // Utiliser la couche d'embedding pour convertir les tokens en embeddings
        INDArray input = tokenizer.lookupEmbeddings(tokens); // [batchSize, seqLength, dModel]

        // Création du masque de padding en utilisant les tokens
        INDArray mask = NDArrayUtils.createKeyPaddingMask(tokenizer, tokens);
        
                
        // Définir le masque attendu
        // [
        //   [
        //     [ [0.0, 0.0, -Infinity, -Infinity] ]
        //   ]
        // ]
        INDArray expectedMask = Nd4j.createFromArray(new float[][][][] {
            {
                {
                    {1f, 1f, 0f, 0f}
                }
            }
        }); // [1, 1, 1, 4]
        
        // Vérifier la forme du masque
        assertArrayEquals(expectedMask.shape(), mask.shape(),
            "La forme du masque de padding n'est pas correcte.");
        
        // Vérifier les valeurs du masque
        for (int b = 0; b < mask.size(0); b++) {
            for (int h = 0; h < mask.size(1); h++) {
                for (int i = 0; i < mask.size(2); i++) {
                    for (int j = 0; j < mask.size(3); j++) {
                        float expectedValue = expectedMask.getFloat(b, h, i, j);
                        float actualValue = mask.getFloat(b, h, i, j);
                        assertEquals(expectedValue, actualValue, 1e-6f,
                        String.format("Position [%d][%d][%d][%d] devrait être %.1f, mais était %.2f",
                            b, h, i, j, expectedValue, actualValue));
                    }
                }
            }
        }
        
        System.out.println("Test réussi : La méthode createPaddingMask génère le masque correct avec des valeurs spécifiques.");
    }




}
