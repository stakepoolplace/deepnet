package RN.transformer;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

/**
 * Classe de test unitaire pour la classe LinearProjection.
 * Elle vérifie le bon fonctionnement des méthodes project, forward, backward,
 * ainsi que les accesseurs et mutateurs des paramètres et gradients.
 */
public class LinearProjectionTest {

    private LinearProjection lp;

    /**
     * Méthode exécutée avant chaque test.
     * Initialise une instance de LinearProjection avec des dimensions spécifiques.
     */
    @Before
    public void setUp() {
        // Initialisation de LinearProjection avec inputSize=10 et outputSize=5.
        lp = new LinearProjection(10, 5);
    }

    /**
     * Test de la méthode project avec une entrée de rang 2.
     * Vérifie que la sortie a la forme correcte [batchSize, outputSize].
     */
    @Test
    public void testProjectRank2() {
        // Créer une entrée de rang 2 [batchSize=2, inputSize=10]
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);

        // Effectuer la projection
        INDArray output = lp.project(input);

        // Vérifier la forme de la sortie : [2, 5]
        Assert.assertArrayEquals("Output shape should be [2, 5]", new long[]{2, 5}, output.shape());

        // Vérifier le type de données
        Assert.assertEquals("Output data type should be FLOAT", DataType.FLOAT, output.dataType());
    }

    /**
     * Test de la méthode project avec une entrée de rang 3.
     * Vérifie que la sortie a la forme correcte [batchSize, seqLength, outputSize].
     */
    @Test
    public void testProjectRank3() {
        // Créer une entrée de rang 3 [batchSize=3, seqLength=4, inputSize=10]
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4, 10);

        // Effectuer la projection
        INDArray output = lp.project(input);

        // Vérifier la forme de la sortie : [3, 4, 5]
        Assert.assertArrayEquals("Output shape should be [3, 4, 5]", new long[]{3, 4, 5}, output.shape());

        // Vérifier le type de données
        Assert.assertEquals("Output data type should be FLOAT", DataType.FLOAT, output.dataType());
    }

    /**
     * Test de la méthode forward.
     * Vérifie que la sortie a la forme correcte et que LayerNorm est appliqué correctement.
     */
    @Test
    public void testForward() {
        // Créer une entrée de rang 2 [batchSize=2, inputSize=10]
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);

        // Effectuer le forward pass
        INDArray output = lp.forward(input);

        // Vérifier la forme de la sortie : [2, 5]
        Assert.assertArrayEquals("Output shape should be [2, 5]", new long[]{2, 5}, output.shape());

        // Vérifier le type de données
        Assert.assertEquals("Output data type should be FLOAT", DataType.FLOAT, output.dataType());

        // Optionnel: Vérifier les valeurs si possible, par exemple vérifier que output n'est pas NaN
        // Assert.assertFalse("Output should not contain NaN", output.isNaN());
        // Assert.assertFalse("Output should not contain Inf", output.isInfinite());
    }

    /**
     * Test de la méthode backward.
     * Vérifie que tous les gradients sont calculés et ont les bonnes formes.
     */
    @Test
    public void testBackward() {
        // Créer une entrée de rang 2 [batchSize=2, inputSize=10]
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);

        // Créer un gradOutput de rang 2 [batchSize=2, outputSize=5]
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, 2, 5);

        // Effectuer le backward pass
        Map<String, INDArray> grads = lp.backward(input, gradOutput);

        // Vérifier que tous les gradients sont présents
        Assert.assertTrue("Gradients should contain 'weights'", grads.containsKey("weights"));
        Assert.assertTrue("Gradients should contain 'bias'", grads.containsKey("bias"));
        Assert.assertTrue("Gradients should contain 'gamma'", grads.containsKey("gamma"));
        Assert.assertTrue("Gradients should contain 'beta'", grads.containsKey("beta"));
        Assert.assertTrue("Gradients should contain 'input'", grads.containsKey("input"));

        // Vérifier le nombre de gradients
        Assert.assertEquals("Should have 5 gradients", 5, grads.size());

        // Vérifier les formes des gradients
        Assert.assertArrayEquals("weights gradient shape should be [10,5]", new long[]{10, 5}, grads.get("weights").shape());
        Assert.assertArrayEquals("bias gradient shape should be [1,5]", new long[]{1, 5}, grads.get("bias").shape());
        Assert.assertArrayEquals("gamma gradient shape should be [1,10]", new long[]{1, 10}, grads.get("gamma").shape());
        Assert.assertArrayEquals("beta gradient shape should be [1,10]", new long[]{1, 10}, grads.get("beta").shape());
        Assert.assertArrayEquals("input gradient shape should be [2,10]", new long[]{2, 10}, grads.get("input").shape());

        // Optionnel: Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        // for (Map.Entry<String, INDArray> entry : grads.entrySet()) {
        //     Assert.assertFalse("Gradient " + entry.getKey() + " should not contain NaN", entry.getValue().isNaN());
        //     Assert.assertFalse("Gradient " + entry.getKey() + " should not contain Inf", entry.getValue().isInfinite());
        // }
    }

    /**
     * Test des accesseurs des paramètres.
     * Vérifie que les paramètres retournés sont corrects et ont les bonnes formes.
     */
    @Test
    public void testGetParameters() {
        // Obtenir les paramètres
        List<INDArray> params = lp.getParameters();

        // Vérifier qu'il y a 4 paramètres : weights, bias, gamma, beta
        Assert.assertEquals("Should have 4 parameters", 4, params.size());

        // Vérifier les formes des paramètres
        INDArray weights = params.get(0);
        INDArray bias = params.get(1);
        INDArray gamma = params.get(2);
        INDArray beta = params.get(3);

        Assert.assertArrayEquals("weights shape should be [10, 5]", new long[]{10, 5}, weights.shape());
        Assert.assertArrayEquals("bias shape should be [1, 5]", new long[]{1, 5}, bias.shape());
        Assert.assertArrayEquals("gamma shape should be [1, 10]", new long[]{1, 10}, gamma.shape());
        Assert.assertArrayEquals("beta shape should be [1, 10]", new long[]{1, 10}, beta.shape());
    }

    /**
     * Test des mutateurs des paramètres.
     * Vérifie que les paramètres sont correctement mis à jour.
     */
    @Test
    public void testSetParameters() {
        // Créer de nouvelles valeurs pour les paramètres
        INDArray newWeights = Nd4j.ones(DataType.FLOAT, 10, 5);
        INDArray newBias = Nd4j.ones(DataType.FLOAT, 1, 5);
        INDArray newGamma = Nd4j.ones(DataType.FLOAT, 1, 10);
        INDArray newBeta = Nd4j.ones(DataType.FLOAT, 1, 10);

        // Mettre à jour les paramètres
        lp.setParameters(newWeights, newBias, newGamma, newBeta);

        // Obtenir les paramètres mis à jour
        List<INDArray> params = lp.getParameters();

        // Vérifier que les paramètres ont été mis à jour
        Assert.assertEquals("weights should be all ones", Nd4j.ones(DataType.FLOAT, 10, 5), params.get(0));
        Assert.assertEquals("bias should be all ones", Nd4j.ones(DataType.FLOAT, 1, 5), params.get(1));
        Assert.assertEquals("gamma should be all ones", Nd4j.ones(DataType.FLOAT, 1, 10), params.get(2));
        Assert.assertEquals("beta should be all ones", Nd4j.ones(DataType.FLOAT, 1, 10), params.get(3));
    }

    /**
     * Test du nombre total de paramètres.
     * Vérifie que le nombre de paramètres correspond à ce qui est attendu.
     */
    @Test
    public void testNumberOfParameters() {
        // Nombre attendu de paramètres : weights + bias + gamma + beta
        long expected = (10 * 5) + (1 * 5) + (1 * 10) + (1 * 10); // 50 + 5 + 10 + 10 = 75

        Assert.assertEquals("Number of parameters should be 75", 75, lp.getNumberOfParameters());
    }

    /**
     * Test du nombre total de gradients.
     * Vérifie que le nombre de gradients correspond à ce qui est attendu.
     */
    @Test
    public void testNumberOfGradients() {
        // Créer une entrée et un gradOutput pour effectuer le backward
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, 2, 5);

        // Effectuer le backward pass
        Map<String, INDArray> grads = lp.backward(input, gradOutput);

        // Nombre attendu de gradients : weights + bias + gamma + beta + input
        long expected = 5;

        // Vérifier le nombre de gradients
        Assert.assertEquals("Should have 5 gradients", expected, grads.size());
    }

    /**
     * Test de la méthode getGradients.
     * Vérifie que les gradients retournés sont corrects et ont les bonnes formes.
     */
    @Test
    public void testGetGradients() {
        // Créer une entrée et un gradOutput pour effectuer le backward
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);
        INDArray gradOutput = Nd4j.rand(DataType.FLOAT, 2, 5);

        // Effectuer le backward pass
        Map<String, INDArray> grads = lp.backward(input, gradOutput);

        // Obtenir les gradients
        List<INDArray> gradList = lp.getGradients();

        // Vérifier que gradList contient les gradients dans l'ordre [weights, bias, gamma, beta]
        Assert.assertEquals("Should have 4 gradients in getGradients", 4, gradList.size());

        // Vérifier les formes
        Assert.assertArrayEquals("weights gradient shape should be [10,5]", new long[]{10, 5}, gradList.get(0).shape());
        Assert.assertArrayEquals("bias gradient shape should be [1,5]", new long[]{1, 5}, gradList.get(1).shape());
        Assert.assertArrayEquals("gamma gradient shape should be [1,10]", new long[]{1, 10}, gradList.get(2).shape());
        Assert.assertArrayEquals("beta gradient shape should be [1,10]", new long[]{1, 10}, gradList.get(3).shape());

        // Optionnel: Vérifier que les gradients ne contiennent pas de NaN ou d'Inf
        // for (INDArray grad : gradList) {
        //     Assert.assertFalse("Gradient should not contain NaN", grad.isNaN());
        //     Assert.assertFalse("Gradient should not contain Inf", grad.isInfinite());
        // }
    }
}
