package RN.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class PositionwiseFeedForward {
	private INDArray W1, b1, W2, b2;
	private INDArray inputCache, reluCache; // Cache pour le forward
	private Map<String, INDArray> gradients = new HashMap<>();

	public PositionwiseFeedForward(int modelSize, int ffSize) {
		this.W1 = Nd4j.rand(modelSize, ffSize);
		this.b1 = Nd4j.rand(1, ffSize);
		this.W2 = Nd4j.rand(ffSize, modelSize);
		this.b2 = Nd4j.rand(1, modelSize);
	}

	public INDArray forward(INDArray input) {
		this.inputCache = input.dup();
		INDArray hidden = input.mmul(W1).addRowVector(b1);
		this.reluCache = hidden.dup();
		INDArray output = Transforms.relu(hidden).mmul(W2).addRowVector(b2);
		return output;
	}

	public Map<String, INDArray> backward(INDArray gradOutput) {
	    // Utilisation de reluCache pour déterminer où la sortie était > 0
	    INDArray reluGrad = this.reluCache.gt(0); // 1 pour les éléments > 0, sinon 0
	    INDArray gradThroughRelu = gradOutput.mul(reluGrad); // Application de la dérivée de ReLU
	    
	    // Calcul des gradients par rapport à W2 et b2
	    INDArray gradW2 = this.reluCache.transpose().mmul(gradThroughRelu);
	    INDArray gradB2 = gradThroughRelu.sum(0);
	    
	    // Propagation du gradient à travers la deuxième couche linéaire
	    INDArray gradHidden = gradThroughRelu.mmul(W2.transpose());
	    
	    // Calcul des gradients par rapport à W1 et b1
	    INDArray gradW1 = this.inputCache.transpose().mmul(gradHidden);
	    INDArray gradB1 = gradHidden.sum(0);
	    
	    // Calcul du gradient à propager à la couche précédente
	    INDArray gradInput = gradHidden.mmul(W1.transpose());
	    
	    gradients.put("W1", gradW1);
	    gradients.put("b1", gradB1);
	    gradients.put("W2", gradW2);
	    gradients.put("b2", gradB2);
	    gradients.put("input", gradInput);

	    return gradients;
	}


	public List<INDArray> getParameters() {
	    // Inclure les biais dans la liste des paramètres retournés
	    return Arrays.asList(W1, b1, W2, b2);
	}
	
	public List<INDArray> getGradients() {
		return Arrays.asList(gradients.get("W1"), gradients.get("b1"), gradients.get("W2"), gradients.get("b2"));
	}


	public long getNumberOfParameters() {
	    // Calculer le total en incluant aussi les éléments des vecteurs de biais
	    return W1.length() + b1.length() + W2.length() + b2.length();
	}
	
	public long getNumberOfGradients() {
	    return gradients.get("W1").length() + gradients.get("b1").length() + gradients.get("W2").length() + gradients.get("b2").length();
	}

}
