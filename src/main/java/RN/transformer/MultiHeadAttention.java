package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MultiHeadAttention {
	
    private int dModel;
    private int numHeads;
    private INDArray Wq, Wk, Wv, Wo; // Poids pour les requêtes, clés, valeurs et sortie

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
        INDArray attentionOutput = attentionWeights.mmul(v);

        // Projection de la sortie de l'attention multi-têtes
        return attentionOutput.mmul(Wo);
    }
    
    public List<INDArray> getParameters() {
        // Retourner les matrices de poids comme une liste d'INDArray
        return Arrays.asList(Wq, Wk, Wv, Wo);
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
