package RN.transformer;


import java.util.stream.IntStream;

import org.apache.commons.math3.linear.*;

public class Attention {
    private int numHeads;
    private int modelSize;
    
    public Attention(int numHeads, int modelSize) {
        this.numHeads = numHeads;
        this.modelSize = modelSize;
    }
    
    public RealMatrix multiHeadAttention(RealMatrix queries, RealMatrix keys, RealMatrix values) {
        int headDim = modelSize / numHeads;
        RealMatrix[] heads = new RealMatrix[numHeads];
        
        IntStream.range(0, numHeads).parallel().forEach(i -> {
            RealMatrix q = queries.getSubMatrix(0, queries.getRowDimension() - 1, i * headDim, (i + 1) * headDim - 1);
            RealMatrix k = keys.getSubMatrix(0, keys.getRowDimension() - 1, i * headDim, (i + 1) * headDim - 1);
            RealMatrix v = values.getSubMatrix(0, values.getRowDimension() - 1, i * headDim, (i + 1) * headDim - 1);
            heads[i] = scaledDotProductAttention(q, k, v);
        });
        
        // Concatenate the heads (rows side by side)
        RealMatrix output = null; //MatrixUtils.createRealMatrix(queries.getRowDimension(), modelSize);
        for (int i = 0; i < numHeads; i++) {
            output.setSubMatrix(heads[i].getData(), 0, i * headDim);
        }
        
        return output;
    }
    
    private RealMatrix scaledDotProductAttention(RealMatrix q, RealMatrix k, RealMatrix v) {
        double sqrtDk = Math.sqrt(k.getColumnDimension());
        RealMatrix scores = q.multiply(k.transpose()).scalarMultiply(1.0 / sqrtDk);
        RealMatrix weights = softmax(scores);
        return weights.multiply(v);
    }
    
    private RealMatrix softmax(RealMatrix matrix) {
        // Implement softmax function across the rows for each column.
        // This is a placeholder implementation.
        return matrix; // Placeholder
    }
}
