package RN.transformer;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearProjectionTest {

    private LinearProjection lp;

    @Before
    public void setUp() {
        // Initialize LinearProjection with input and output sizes.
        // This example assumes such a constructor exists.
    	try {
            lp = new LinearProjection(10, 5);
    	} catch(Throwable t) {
    		t.printStackTrace();
    	}
    			
    	
    }

    @Test
    public void testProjectOutputSize() {
        // Create a dummy input INDArray with the shape [batchSize, inputSize].
        // Here, batchSize is 1 for simplicity, and inputSize is 10.
        INDArray input = Nd4j.zeros(1, 10);
        
        // Perform the projection.
        INDArray output = lp.project(input);

        // Check that the output has the correct shape: [batchSize, outputSize].
        Assert.assertEquals("Output size should be 5", 5, output.size(1));
    }

    // Additional tests as necessary...
}
