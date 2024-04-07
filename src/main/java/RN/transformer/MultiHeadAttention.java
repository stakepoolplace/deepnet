package RN.transformer;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MultiHeadAttention {
	
    private int dModel;
    private int numHeads;
    private INDArray inputQ, inputK, inputV; // Inputs cachés pour le backward
    private INDArray Wq, Wk, Wv, Wo; // Poids pour les requêtes, clés, valeurs et sortie
    private INDArray attentionWeights; // Poids d'attention cachés pour le backward
    private INDArray attentionOutput;
    private Map<String, INDArray> gradients = new HashMap<>();
    
    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        // Initialisation des poids. Les dimensions réelles dépendent de l'architecture spécifique.
        Wq = Nd4j.rand(dModel, dModel);
        Wk = Nd4j.rand(dModel, dModel);
        Wv = Nd4j.rand(dModel, dModel);
        Wo = Nd4j.rand(dModel, dModel);
    }

    public INDArray forward(INDArray query, INDArray key, INDArray value, INDArray mask) {
        
    	// Cacher les inputs pour utilisation dans backward
        this.inputQ = query.dup();
        this.inputK = key.dup();
        this.inputV = value.dup();
    	
    	INDArray q = query.mmul(Wq);
        INDArray k = key.mmul(Wk);
        INDArray v = value.mmul(Wv);

        // Calcul des scores d'attention
        INDArray attentionScores = q.mmul(k.transpose()).div(Math.sqrt(dModel / numHeads));

        // Application du masque, si fourni
        if (mask != null) {
            // Les éléments du masque avec de très grandes valeurs négatives deviennent zéro après softmax
            attentionScores.addi(mask);
        }

        INDArray attentionWeights = Transforms.softmax(attentionScores); // Softmax sur la dernière dimension
        this.attentionWeights = attentionWeights;


        attentionOutput = attentionWeights.mmul(v);
        	
        // Projection de la sortie de l'attention multi-têtes
        return attentionOutput.mmul(Wo);
    }
    
    
    // Supposons une méthode forward qui initialise correctement inputQ, inputK, inputV, et attentionWeights
    
    public Map<String, INDArray> backward(INDArray gradOutput) {

        
        // Calcul simplifié du gradient par rapport aux scores d'attention
        INDArray gradAttention = gradOutput.mmul(this.Wv.transpose());
        
        // Gradient par rapport aux entrées Q et K (simplifié)
        INDArray gradQ = gradAttention.mul(inputK.transpose());
        INDArray gradK = gradAttention.transpose().mul(inputQ);
        // Calcul du gradient par rapport à la sortie de la valeur V
        INDArray gradV = attentionWeights.transpose().mmul(gradOutput);
        
        // Gradient par rapport aux poids Wq et Wk
        INDArray gradWq = inputQ.transpose().mmul(gradQ);
        INDArray gradWk = inputK.transpose().mmul(gradK);
        INDArray gradWv = inputV.transpose().mmul(gradV);

        // Calcul du gradient par rapport à la sortie de la projection finale
        INDArray gradWo = attentionOutput.transpose().mmul(gradOutput);
        
        gradients.put("Wq", gradWq);
        gradients.put("Wk", gradWk);
        gradients.put("Wv", gradWv);
        gradients.put("Wo", gradWo);
        
        // Supposant que les gradients par rapport aux entrées sont nécessaires pour la rétropropagation à travers le réseau
        gradients.put("inputQ", gradQ);
        gradients.put("inputK", gradK);
        gradients.put("inputV", gradV);
        
        return gradients;
    }  
    
    
    
    public List<INDArray> getParameters() {
        // Retourner les matrices de poids comme une liste d'INDArray
        return Arrays.asList(Wq, Wk, Wv, Wo);
    }
    
	public List<INDArray> getGradients() {
		return Arrays.asList(gradients.get("Wq"), gradients.get("Wk"), gradients.get("Wv"), gradients.get("Wo"));
	}
    
    public long getNumberOfParameters() {
    	return Wq.length() + Wk.length() + Wv.length() + Wo.length();
    }
    
    public long getNumberOfGradients() {
    	return gradients.get("Wq").length() + gradients.get("Wk").length() + gradients.get("Wv").length() + gradients.get("Wo").length();
    }

	public int getdModel() {
		return dModel;
	}

	public void setdModel(int dModel) {
		this.dModel = dModel;
	}

	public int getNumHeads() {
		return numHeads;
	}

	public void setNumHeads(int numHeads) {
		this.numHeads = numHeads;
	}

	public INDArray getWq() {
		return Wq;
	}

	public void setWq(INDArray wq) {
		Wq = wq;
	}

	public INDArray getWk() {
		return Wk;
	}

	public void setWk(INDArray wk) {
		Wk = wk;
	}

	public INDArray getWv() {
		return Wv;
	}

	public void setWv(INDArray wv) {
		Wv = wv;
	}

	public INDArray getWo() {
		return Wo;
	}

	public void setWo(INDArray wo) {
		Wo = wo;
	}




}
