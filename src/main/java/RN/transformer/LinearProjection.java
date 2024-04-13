package RN.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LinearProjection {
	
    private INDArray weights, bias;
    private INDArray inputCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();
    private double epsilon = 1e-7; // Small constant to avoid division by zero
    private INDArray gamma;
    private INDArray beta;
    
    public LinearProjection(int inputSize, int outputSize) {
        // Initialize weights with a normal distribution divided by sqrt(inputSize) for He initialization
        this.weights = Nd4j.randn(inputSize, outputSize).divi(Math.sqrt(inputSize));
        // Initialize biases to zeros for standard practice
        this.bias = Nd4j.zeros(1, outputSize);
        // Initialize gamma to ones and beta to zeros for normalization scaling and shifting
        this.gamma = Nd4j.ones(outputSize); 
        this.beta = Nd4j.zeros(outputSize); 
    }



    public INDArray project(INDArray input) {
        // Projection linéaire en multipliant l'entrée par les poids
        return input.mmul(weights);
    }


    
    public INDArray forward(INDArray input) {
        this.inputCache = input.dup(); // Duplicate the input to avoid mutable changes
        INDArray normalized = normalize(input);
        INDArray output = normalized.mul(gamma).add(beta); // Scale and shift
        return output;
    }

    private INDArray normalize(INDArray input) {
        INDArray mean = input.mean(1);
        INDArray variance = input.var(false, 1);
        return input.sub(mean).div(Transforms.sqrt(variance.add(epsilon)));
    }
    
    public Map<String, INDArray> backward(INDArray gradOutput) {
        if (inputCache == null) {
            throw new IllegalStateException("inputCache is not set. Ensure forward pass is called before backward.");
        }

        long N = inputCache.shape()[1];
        INDArray inputMu = inputCache.sub(inputCache.mean(1));
        INDArray stdInv = Transforms.pow(inputCache.var(false, 1).add(epsilon), -0.5);

        INDArray gradInput = gradOutput.mul(gamma).mul(stdInv);
        INDArray gradGamma = gradOutput.mul(inputMu).mul(stdInv).sum(0);
        INDArray gradBeta = gradOutput.sum(0);

        Map<String, INDArray> gradients = new HashMap<>();
        gradients.put("input", gradInput);
        gradients.put("gamma", gradGamma);
        gradients.put("beta", gradBeta);

        return gradients;
    }


    // Méthode pour obtenir les paramètres (poids) de la projection
    public List<INDArray> getParameters() {
        return Arrays.asList(weights, bias);
    }
    
	public List<INDArray> getGradients() {
		return Arrays.asList(gradients.get("weights"), gradients.get("bias"));
	}

    // Méthode pour définir (mettre à jour) les paramètres (poids) de la projection
    public void setParameters(INDArray newWeights) {
        this.weights = newWeights;
    }

	public long getNumberOfParameters() {
		return weights.length() + bias.length();
	}
	
	public long getNumberOfGradients() {
		return gradients.get("weights").length() + gradients.get("bias").length();
	}	
}
