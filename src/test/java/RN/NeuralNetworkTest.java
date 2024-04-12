package RN;

import java.util.Arrays;

import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {
    NeuralNetwork nn;

    @Before
    public void setUp() {
        nn = new NeuralNetwork();
    }

    @Test
    public void testLearningProcess() {
        double[] inputs = {1.0, 0.0}; // Simuler des entrées binaires
        double[] expected = {1.0, 0.0}; // Simuler la sortie attendue pour la classification

        for (int i = 0; i < 10; i++) {
            double[] outputs = nn.forwardPass(inputs);
            nn.backwardPass(expected);
            System.out.println("Epoch " + i + ", Output: " + outputs[0]);
            // Afficher les poids pour vérifier les ajustements
            System.out.println("Weights Output: " + Arrays.toString(nn.getWeightsOutput()));
            System.out.println("Weights Hidden: " + Arrays.deepToString(nn.getWeightsHidden()));
        }
        
        // Ajouter des assertions pour vérifier les valeurs attendues après des epochs spécifiques si nécessaire
    }
}
