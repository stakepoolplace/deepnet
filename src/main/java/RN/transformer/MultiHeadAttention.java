package RN.transformer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MultiHeadAttention implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7153801764592720027L;
	private int dModel;
	private int numHeads;
	private int depth;
	private INDArray inputQ, inputK, inputV; // Inputs cachés pour le backward
	private INDArray Wq, Wk, Wv, Wo; // Poids pour les requêtes, clés, valeurs et sortie
	private INDArray attentionWeights; // Poids d'attention cachés pour le backward
	private INDArray attentionOutput;
	private Map<String, INDArray> gradients = new HashMap<>();

	public MultiHeadAttention(int dModel, int numHeads) {
		if (dModel % numHeads != 0) {
			throw new IllegalArgumentException("dModel must be divisible by numHeads");
		}
		this.dModel = dModel;
		this.numHeads = numHeads;
		this.depth = dModel / numHeads;
		// Initialisation des poids. Les dimensions réelles dépendent de l'architecture
		// spécifique.
		Wq = Nd4j.rand(dModel, numHeads * depth);
		Wk = Nd4j.rand(dModel, numHeads * depth);
		Wv = Nd4j.rand(dModel, numHeads * depth);
		Wo = Nd4j.rand(dModel, dModel);
	}

	public INDArray forward(INDArray query, INDArray key, INDArray value, INDArray mask) {
		// Cacher les inputs pour une utilisation dans backward
		this.inputQ = query.dup();
		this.inputK = key.dup();
		this.inputV = value.dup();

		// Trouver la longueur maximale des séquences dans le lot
		int sequenceLength = (int) Math.max(query.shape()[0], Math.max(key.shape()[0], value.shape()[0]));

		// Remplir (padding) les séquences plus courtes avec des zéros (si nécessaire)
		query = padSequence(query, sequenceLength);
		key = padSequence(key, sequenceLength);
		value = padSequence(value, sequenceLength);

		// Projection linéaire des requêtes, clés et valeurs
		INDArray q = query.mmul(Wq).reshape(sequenceLength, numHeads, depth).permute(1, 0, 2); // [numHeads, seqLength,
																								// depth]
		INDArray k = key.mmul(Wk).reshape(sequenceLength, numHeads, depth).permute(1, 0, 2); // [numHeads, seqLength,
																								// depth]
		INDArray v = value.mmul(Wv).reshape(sequenceLength, numHeads, depth).permute(1, 0, 2); // [numHeads, seqLength,
																								// depth]

		// Transposer k pour obtenir [numHeads, depth, seqLength]
		k = k.permute(0, 2, 1);

		// Calcul des scores d'attention (produit matriciel Q x K^T)
		INDArray attentionScores = Nd4j.matmul(q, k).div(Math.sqrt(depth)); // [numHeads, seqLength, seqLength]

		// Application du masque, si fourni
		if (mask != null) {
			attentionScores.addi(mask); // Masque les scores non pertinents
		}

		// Calcul des poids d'attention avec softmax
		INDArray attentionWeights = Transforms.softmax(attentionScores);
		this.attentionWeights = attentionWeights;

		// Calcul de l'output avec V
		INDArray attentionOutput = Nd4j.matmul(attentionWeights, v); // [numHeads, seqLength, depth]

		// Permutation et reshape pour combiner les têtes
		this.attentionOutput = attentionOutput.permute(1, 0, 2) // [seqLength, numHeads, depth]
				.reshape(sequenceLength, numHeads * depth); // [seqLength, numHeads * depth]

		// Appliquer Wo pour la transformation linéaire finale
		return this.attentionOutput.mmul(Wo); // [seqLength, dModel]
	}

	private INDArray padSequence(INDArray sequence, int maxSeqLength) {
		int batchSize = (int) sequence.shape()[0];
		int seqLength = (int) sequence.shape()[1];

		if (seqLength < maxSeqLength) {
			INDArray paddingTensor = Nd4j.zeros(batchSize, maxSeqLength - seqLength, dModel);
			sequence = Nd4j.hstack(sequence, paddingTensor);
		}

		return sequence;
	}

	public Map<String, INDArray> backward(INDArray gradOutput) {
	    if (this.attentionOutput == null) {
	        throw new IllegalStateException("attentionOutput est null. Assurez-vous d'appeler la méthode forward avant backward.");
	    }

	    // 1. Définir les variables nécessaires
	    int seqLength = (int) attentionOutput.shape()[0];
	    int numHeads = this.numHeads;
	    int depth = this.depth;

	    // 2. Calcul du gradient par rapport à Wo
	    INDArray gradWo = attentionOutput.transpose().mmul(gradOutput); // [numHeads * depth, dModel]
	    gradients.put("Wo", gradWo);

	    // 3. Calcul du gradient par rapport à attentionOutput
	    INDArray gradAttentionOutput = gradOutput.mmul(Wo.transpose()); // [seqLength, numHeads * depth]

	    // 4. Reshape gradAttentionOutput de [seqLength, numHeads * depth] à [numHeads, seqLength, depth]
	    INDArray gradAttentionOutputReshaped = gradAttentionOutput.reshape(numHeads, seqLength, depth); // [numHeads, seqLength, depth]

	    // 5. Calcul de gradV = attentionWeights [numHeads, seqLength, seqLength] mmul gradAttentionOutputReshaped [numHeads, seqLength, depth] = [numHeads, seqLength, depth]
	    INDArray gradV = Nd4j.create(numHeads, seqLength, depth); // [numHeads, seqLength, depth]
	    for (int h = 0; h < numHeads; h++) {
	        // Remplacer getRow par slice
	        INDArray attentionWeightsHead = attentionWeights.slice(h); // [seqLength, seqLength]
	        
	        // Remplacer getRow par slice
	        INDArray gradAttentionOutputHead = gradAttentionOutputReshaped.slice(h); // [seqLength, depth]
	        
	        // Multiplication matricielle : [seqLength, seqLength] mmul [seqLength, depth] = [seqLength, depth]
	        INDArray gradVHead = attentionWeightsHead.mmul(gradAttentionOutputHead); // [seqLength, depth]
	        
	        // Remplacer putRow par putSlice
	        gradV.putSlice(h, gradVHead); // Assigner gradVHead à gradV[h]
	    }

	    // 6. Reshape gradV de [numHeads, seqLength, depth] à [seqLength, numHeads * depth]
	    INDArray gradVReshaped = gradV.reshape(seqLength, numHeads * depth); // [seqLength, numHeads * depth]

	    // 7. Calcul de gradWv
	    INDArray gradWv = inputV.transpose().mmul(gradVReshaped); // [dModel, numHeads * depth]
	    gradients.put("Wv", gradWv);

	    // 8. Reshape V de [seqLength, numHeads * depth] à [numHeads, seqLength, depth]
	    INDArray V = inputV.mmul(Wv); // [seqLength, numHeads * depth]
	    INDArray VReshaped = V.reshape(numHeads, seqLength, depth); // [numHeads, seqLength, depth]

	    // 9. Calcul de gradScores = gradAttentionOutputReshaped [numHeads, seqLength, depth] mmul VReshaped.transpose() [numHeads, depth, seqLength] = [numHeads, seqLength, seqLength]
	    INDArray gradScores = Nd4j.create(numHeads, seqLength, seqLength); // [numHeads, seqLength, seqLength]
	    for (int h = 0; h < numHeads; h++) {
	        // VReshaped[h].transpose() : [depth, seqLength]
	        INDArray VhT = VReshaped.slice(h).transpose(); // [depth, seqLength]

	        // gradAttentionOutputReshaped[h] : [seqLength, depth]
	        INDArray gradAttentionOutputHead = gradAttentionOutputReshaped.slice(h); // [seqLength, depth]

	        // Calcul de gradScores[h] : [seqLength, depth] mmul [depth, seqLength] = [seqLength, seqLength]
	        INDArray gradScoresHead = gradAttentionOutputHead.mmul(VhT); // [seqLength, seqLength]

	        // Remplacer putRow par putSlice
	        gradScores.putSlice(h, gradScoresHead); // Assigner gradScoresHead à gradScores[h]
	    }

	    // 10. Calcul du gradient de la softmax
	    INDArray gradAttentionScoresFinal = softmaxGrad(attentionWeights, gradScores); // [numHeads, seqLength, seqLength]

	    // 11. Calcul de Q : [seqLength, numHeads * depth] = [seqLength, numHeads * depth]
	    INDArray Q = inputQ.mmul(Wq); // [seqLength, numHeads * depth]
	    INDArray QReshaped = Q.reshape(numHeads, seqLength, depth); // [numHeads, seqLength, depth]

	    // 12. Calcul de gradQ = gradScoresFinal [numHeads, seqLength, seqLength] mmul (inputK * Wk) [numHeads, seqLength, depth] = [numHeads, seqLength, depth]
	    INDArray inputK_proj = inputK.mmul(Wk); // [seqLength, numHeads * depth]
	    INDArray inputK_projReshaped = inputK_proj.reshape(numHeads, seqLength, depth); // [numHeads, seqLength, depth]
	    INDArray gradQ_full = Nd4j.create(numHeads, seqLength, depth); // [numHeads, seqLength, depth]
	    for (int h = 0; h < numHeads; h++) {
	        // gradAttentionScoresFinal[h] : [seqLength, seqLength]
	        INDArray gradScoresHead = gradAttentionScoresFinal.slice(h); // [seqLength, seqLength]
	        
	        // inputK_projReshaped[h] : [seqLength, depth]
	        INDArray inputK_projHead = inputK_projReshaped.slice(h); // [seqLength, depth]
	        
	        // Calcul de gradQ_full[h] : [seqLength, depth] = [seqLength, seqLength] mmul [seqLength, depth]
	        INDArray gradQHead = gradScoresHead.mmul(inputK_projHead); // [seqLength, depth]
	        
	        // Remplacer putRow par putSlice
	        gradQ_full.putSlice(h, gradQHead); // Assigner gradQHead à gradQ_full[h]
	    }

	    // Reshape gradQ_full de [numHeads, seqLength, depth] à [seqLength, numHeads * depth]
	    INDArray gradQReshaped = gradQ_full.reshape(seqLength, numHeads * depth); // [seqLength, numHeads * depth]

	    // 13. Calcul de gradK = gradScoresFinal.transpose() [numHeads, seqLength, seqLength] mmul Q [numHeads, seqLength, depth] = [numHeads, seqLength, depth]
	    INDArray gradK_full = Nd4j.create(numHeads, seqLength, depth); // [numHeads, seqLength, depth]
	    for (int h = 0; h < numHeads; h++) {
	        // gradAttentionScoresFinal[h].transpose() : [seqLength, seqLength]
	        INDArray gradScoresHeadTrans = gradAttentionScoresFinal.slice(h).transpose(); // [seqLength, seqLength]
	        
	        // QReshaped[h] : [seqLength, depth]
	        INDArray QHead = QReshaped.slice(h); // [seqLength, depth]
	        
	        // Calcul de gradK_full[h] : [seqLength, depth] = [seqLength, seqLength] mmul [seqLength, depth]
	        INDArray gradKHead = gradScoresHeadTrans.mmul(QHead); // [seqLength, depth]
	        
	        // Remplacer putRow par putSlice
	        gradK_full.putSlice(h, gradKHead); // Assigner gradKHead à gradK_full[h]
	    }

	    // Reshape gradK_full de [numHeads, seqLength, depth] à [seqLength, numHeads * depth]
	    INDArray gradKReshaped = gradK_full.reshape(seqLength, numHeads * depth); // [seqLength, numHeads * depth]

	    // 14. Calcul de gradWq et gradWk
	    INDArray gradWq = inputQ.transpose().mmul(gradQReshaped); // [dModel, numHeads * depth]
	    INDArray gradWk = inputK.transpose().mmul(gradKReshaped); // [dModel, numHeads * depth]
	    gradients.put("Wq", gradWq);
	    gradients.put("Wk", gradWk);

	    // 15. Calcul des gradients par rapport aux entrées Q, K, V
	    INDArray gradInputQ = gradQReshaped.mmul(Wq.transpose()); // [seqLength, dModel]
	    INDArray gradInputK = gradKReshaped.mmul(Wk.transpose()); // [seqLength, dModel]
	    INDArray gradInputV = gradVReshaped.mmul(Wv.transpose()); // [seqLength, dModel]

	    // Ajouter les gradients spécifiques
	    gradients.put("inputQ", gradInputQ);
	    gradients.put("inputK", gradInputK);
	    gradients.put("inputV", gradInputV);

	    // 16. Calcul du gradient à propager vers les couches précédentes
	    INDArray gradInput = gradInputQ.add(gradInputK).add(gradInputV); // [seqLength, dModel]
	    gradients.put("input", gradInput);

	    return gradients;
	}


	/**
	 * Calcule le gradient de la softmax.
	 *
	 * @param softmax Résultats de la softmax de forme [numHeads, seqLength, seqLength]
	 * @param gradA   Gradients provenant de la couche suivante de la même forme que softmax
	 * @return Gradient par rapport aux scores d'attention de la même forme que softmax
	 */
	private INDArray softmaxGrad(INDArray softmax, INDArray gradA) {
	    // softmax: [numHeads, seqLength, seqLength]
	    // gradA: [numHeads, seqLength, seqLength]

	    // Calcul de dL/dS = softmax * (gradA - sum(gradA * softmax, axis=2, keepdims=true))
	    // ND4J M2.1 ne supporte pas directement les opérations sur les axes multiples, donc effectuer itération manuelle

	    INDArray gradS = Nd4j.create(softmax.shape());

	    int numHeads = (int) softmax.shape()[0];
	    int seqLength = (int) softmax.shape()[1];
	    int seqLength2 = (int) softmax.shape()[2];

	    for (int h = 0; h < numHeads; h++) {
	        for (int i = 0; i < seqLength; i++) {
	            double sum = 0.0;
	            for (int j = 0; j < seqLength2; j++) {
	                sum += softmax.getDouble(h, i, j) * gradA.getDouble(h, i, j);
	            }
	            for (int j = 0; j < seqLength2; j++) {
	                double grad = softmax.getDouble(h, i, j) * (gradA.getDouble(h, i, j) - sum);
	                gradS.putScalar(new int[]{h, i, j}, grad);
	            }
	        }
	    }

	    return gradS;
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
		return gradients.get("Wq").length() + gradients.get("Wk").length() + gradients.get("Wv").length()
				+ gradients.get("Wo").length();
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