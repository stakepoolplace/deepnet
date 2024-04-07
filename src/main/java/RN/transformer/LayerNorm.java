package RN.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LayerNorm extends Layer{
    private INDArray gamma, beta;
    private double epsilon = 1e-6;
    private INDArray inputCache; // Cache pour le forward
    Map<String, INDArray> gradients = new HashMap<>();


    public LayerNorm(int dModel) {
        gamma = Nd4j.ones(dModel);
        beta = Nd4j.zeros(dModel);
    }

    @Override
    public INDArray forward(INDArray x) {
        // Check for NaN or Inf in the input
        if (x.isNaN().any() || x.isInfinite().any()) {
            throw new RuntimeException("LayerNorm.forward received NaN or Infinite values in input.");
        }
        this.inputCache = x.dup();

    	

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
    
    @Override
    public Map<String, INDArray> backward(INDArray gradOutput) {
        INDArray input = this.inputCache;
        long N = input.shape()[1];

        INDArray inputMu = input.sub(input.mean(1));
        INDArray stdInv = Transforms.pow(input.var(false, 1).add(epsilon), -0.5);

        INDArray gradInput = gradOutput.mul(gamma).mul(stdInv);
        INDArray gradGamma = gradOutput.mul(inputMu).mul(stdInv).sum(0);
        INDArray gradBeta = gradOutput.sum(0);

        gradients.put("input", gradInput);
        gradients.put("gamma", gradGamma);
        gradients.put("beta", gradBeta);

        return gradients;
    }

    public List<INDArray> getParameters() {
        // Retourner une liste contenant les paramètres gamma et beta
        return Arrays.asList(gamma, beta);
    }
    
    public List<INDArray> getGradients() {
        return Arrays.asList(gradients.get("gamma"), gradients.get("beta"));
    }
    
    public long getNumberOfParameters() {
        // Puisque gamma et beta ont chacun une taille de dModel, le total est simplement 2 * dModel
        return gamma.length() + beta.length();
    }
    
    public long getNumberOfGradients() {
        return gradients.get("gamma").length() + gradients.get("beta").length();
    }
}
