package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LayerNorm {
    private INDArray gamma, beta;
    private double epsilon = 1e-6;

    public LayerNorm(int dModel) {
        gamma = Nd4j.ones(dModel);
        beta = Nd4j.zeros(dModel);
    }

    public INDArray forward(INDArray x) {
        // Check for NaN or Inf in the input
        if (x.isNaN().any() || x.isInfinite().any()) {
            throw new RuntimeException("LayerNorm.forward received NaN or Infinite values in input.");
        }
    	

        // Assuming mean and std are initially vectors of shape [6]
        INDArray mean = x.mean(1);
        INDArray std = x.std(1);

        std.addi(epsilon);
        // Check for NaN or Inf in intermediate results
        if (mean.isNaN().any() || mean.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in mean calculation.");
        }
        if (std.isNaN().any() || std.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in standard deviation calculation.");
        }
        

        // Broadcast subtraction and division
        INDArray centered = x.subColumnVector(mean);
        INDArray normed = centered.divColumnVector(std);

        // Scale and shift
        INDArray output = normed.mulRowVector(gamma).addRowVector(beta);


        // Check for NaN or Inf in the output
        if (output.isNaN().any() || output.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values produced by LayerNorm normalization.");
        }
        
        return output;
    }

    public List<INDArray> getParameters() {
        // Retourner une liste contenant les paramètres gamma et beta
        return Arrays.asList(gamma, beta);
    }
    
    public long getNumberOfParameters() {
        // Puisque gamma et beta ont chacun une taille de dModel, le total est simplement 2 * dModel
        return gamma.length() + beta.length();
    }
}
