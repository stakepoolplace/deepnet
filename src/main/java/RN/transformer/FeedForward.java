package RN.transformer;


import org.apache.commons.math3.linear.*;

public class FeedForward {
    private int modelSize;
    private int ffDim;
    
    public FeedForward(int modelSize, int ffDim) {
        this.modelSize = modelSize;
        this.ffDim = ffDim;
    }
    
    public RealMatrix forward(RealMatrix input) {
        // Exemple simple de FFN avec une couche cachée.
        // Placeholder pour l'activation et les poids.
        RealMatrix w1 = MatrixUtils.createRealMatrix(modelSize, ffDim);
        RealMatrix w2 = MatrixUtils.createRealMatrix(ffDim, modelSize);
        RealMatrix hidden = relu(input.multiply(w1));
        return hidden.multiply(w2);
    }
    
    private RealMatrix relu(RealMatrix matrix) {
        RealMatrix reluMatrix = matrix.copy();
        for (int r = 0; r < reluMatrix.getRowDimension(); r++) {
            for (int c = 0; c < reluMatrix.getColumnDimension(); c++) {
                double value = reluMatrix.getEntry(r, c);
                reluMatrix.setEntry(r, c, Math.max(0, value));
            }
        }
        return reluMatrix;
    }
}
