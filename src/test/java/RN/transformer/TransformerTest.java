package RN.transformer;


import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TransformerTest {
    
    private TransformerModel model;

    @BeforeEach
    public void setUp() throws Exception {
        // Initialize your model and any other components here
        model = new TransformerModel(); // Assuming the constructor is available and initializes everything
    }

    @Test
    public void testEncodeDecode() {
        String testInput = "The quick brown fox jumps over the lazy dog";
        INDArray encoded = model.encoder.encode(true, model.tokenizer.tokensToIds(model.tokenizer.tokenize(testInput)));
        assertNotNull(encoded, "Encoded output should not be null.");
        List<INDArray> decoded = model.decoder.decode(encoded);
        assertNotNull(decoded, "Decoded output should not be null.");
    }

    @Test
    public void testBackwardPropagation() {
        INDArray gradOutput = Nd4j.rand(1, model.decoder.dModel);
        Map<String, INDArray> gradients = model.decoder.backward(gradOutput);
        assertNotNull(gradients, "Gradients should not be null.");
        assertFalse(gradients.isEmpty(), "Gradients map should not be empty.");
    }

    @Test
    public void testParameterUpdates() {
        INDArray initialWeights = model.decoder.getParameters().get(0).dup();
        model.train(new DataGenerator("path/to/data", "path/to/target", model.tokenizer, 1, 256));
        INDArray updatedWeights = model.decoder.getParameters().get(0);
        assertNotEquals(initialWeights, updatedWeights, "Weights should be updated after training.");
    }

    @Test
    public void testLossCalculation() {
        List<INDArray> logits = List.of(Nd4j.rand(10, model.vocabSize));
        List<Integer> targetTokenIds = List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        float loss = model.calculateCrossEntropyLossAndGradient(logits, targetTokenIds).getLeft();
        assertTrue(loss >= 0, "Loss should be non-negative.");
    }

    @Test
    public void testInference() {
        String response = model.infer("Hello world");
        assertFalse(response.isEmpty(), "Response should not be empty.");
    }

    @Test
    public void testLossDecrease() {
        // Assuming you have a method to run multiple epochs and return the last loss
        double initialLoss = model.runEpochAndGetLoss();
        double laterLoss = model.runEpochAndGetLoss();
        assertTrue(laterLoss < initialLoss, "Loss should decrease after training for an epoch.");
    }

    @Test
    public void testGradientsNonZero() {
        // Run a single backward step and check gradients
        model.train(new DataGenerator("dummy_path", "dummy_target", model.tokenizer, 1, 10));
        boolean allNonZero = model.decoder.getGradients().stream()
                             .allMatch(g -> !g.isZeroNumber());
        assertTrue(allNonZero, "All gradients should be non-zero after training step.");
    }

    @Test
    public void testOutputRange() {
        // Assuming outputs are probabilities from the last layer
        double[] outputs = model.forwardPassAndGetOutputs(new double[]{0.1, 0.2, 0.7}); // Example input
        for (double output : outputs) {
            assertTrue(output >= 0 && output <= 1, "Each output should be a valid probability.");
        }
        assertEquals(1.0, Arrays.stream(outputs).sum(), 0.01, "Sum of output probabilities should be 1.");
    }
}
