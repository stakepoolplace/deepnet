package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

public class BatchTest {
    
    @Test
    public void testDataAndTargetAssignment() {
        // Préparation des données et du tokenizer simulé
        List<String> data = Arrays.asList("data1", "data2");
        List<String> target = Arrays.asList("target1", "target2");

        // Création d'un Tokenizer fictif (adapté à votre contexte réel de test)
        Tokenizer tokenizer = new Tokenizer(Arrays.asList("data1", "data2", "target1", "target2"), 300, 5);
        
        // Conversion attendue des tokens en INDArrays
        INDArray expectedData = tokenizer.tokensToINDArray(data);
        INDArray expectedTarget = tokenizer.tokensToINDArray(target);

        // Création du Batch
        Batch batch = new Batch(data, target, tokenizer);
        
        // Vérifications pour confirmer l'initialisation correcte
        Assert.assertEquals("Data INDArray should match expected output", expectedData, batch.getData());
        Assert.assertEquals("Target INDArray should match expected output", expectedTarget, batch.getTarget());

        // Vérification que le masque est null dans ce constructeur
        Assert.assertNull("Mask should be null when not provided", batch.getMask());
    }
}
