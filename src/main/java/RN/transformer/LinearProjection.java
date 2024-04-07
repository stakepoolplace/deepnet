package RN.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearProjection {
	
    private INDArray weights, bias;
    private INDArray inputCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();


    public LinearProjection(int inputSize, int outputSize) {
        // Initialisation des poids avec une distribution normale centrée autour de zéro et une variance de 1/sqrt(inputSize)
        this.weights = Nd4j.randn(inputSize, outputSize).divi(Math.sqrt(inputSize));
        this.bias = Nd4j.rand(1, outputSize);
    }

    public INDArray project(INDArray input) {
        // Projection linéaire en multipliant l'entrée par les poids
        return input.mmul(weights);
    }

    public INDArray forward(INDArray input) {
        this.inputCache = input.dup();
        return input.mmul(weights).addRowVector(bias);
    }
    
    public Map<String, INDArray> backward(INDArray gradOutput) {
        INDArray gradInput = gradOutput.mmul(weights.transpose());
        INDArray gradWeights = inputCache.transpose().mmul(gradOutput);
        INDArray gradBias = gradOutput.sum(0);

        gradients.put("input", gradInput);
        gradients.put("weights", gradWeights);
        gradients.put("bias", gradBias);

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
