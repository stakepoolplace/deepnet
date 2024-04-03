package RN.transformer;

import static org.junit.Assert.assertArrayEquals; // Import assertArrayEquals for array comparisons

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

public class PositionalEncodingTest {
    
    @Test
    public void testGetPositionalEncodingShape() {
        int dModel = 512;
        long sequenceLength = 20;
        PositionalEncoding pe = new PositionalEncoding(dModel);
        
        INDArray posEncoding = pe.getPositionalEncoding(sequenceLength);
        
        // Use assertArrayEquals to compare shapes
        assertArrayEquals("The shape of positional encoding should match [sequenceLength, dModel]", 
                          new long[]{sequenceLength, dModel}, 
                          posEncoding.shape());
    }
}
