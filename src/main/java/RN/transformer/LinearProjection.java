package RN.transformer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Classe représentant une projection linéaire avec normalisation de couche (LayerNorm).
 * Les tenseurs sont supposés avoir la forme [seqLength, dModel].
 */
public class LinearProjection implements Serializable {
    
    private static final long serialVersionUID = -6601830517666118676L;
    private INDArray weights; // Poids de la projection linéaire [dModel, outputSize]
    private INDArray bias;    // Biais de la projection linéaire [1, outputSize]
    private INDArray gamma;   // Paramètre de scale pour LayerNorm [1, dModel]
    private INDArray beta;    // Paramètre de shift pour LayerNorm [1, dModel]
    private final double epsilon = 1e-7; // Petite constante pour éviter la division par zéro
    
    // Gradients calculés lors de la passe backward
    private Map<String, INDArray> gradients = new HashMap<>();
    
    /**
     * Constructeur de la classe LinearProjection.
     * 
     * @param inputSize  Taille de l'entrée (dModel)
     * @param outputSize Taille de la sortie (par exemple, la taille du vocabulaire)
     */
    public LinearProjection(int inputSize, int outputSize) {
        // Initialisation des poids avec une distribution normale divisée par sqrt(inputSize) pour l'initialisation de He
        this.weights = Nd4j.randn(DataType.FLOAT, inputSize, outputSize).div(Math.sqrt(inputSize));
        // Initialisation des biais à zéro
        this.bias = Nd4j.zeros(DataType.FLOAT,1, outputSize);
        // Initialisation de gamma à un et beta à zéro pour la normalisation de couche
        this.gamma = Nd4j.ones(DataType.FLOAT,1, inputSize); // [1, dModel]
        this.beta = Nd4j.zeros(DataType.FLOAT,1, inputSize); // [1, dModel]
    }

    /**
     * Effectue la projection linéaire.
     * 
     * @param input Entrée de forme [seqLength, dModel]
     * @return Sortie projetée de forme [seqLength, outputSize]
     */
    public INDArray project(INDArray input) {
        // Projection linéaire: [seqLength, dModel] * [dModel, outputSize] + [1, outputSize] = [seqLength, outputSize]
        return input.mmul(weights).addRowVector(bias);
    }

    /**
     * Passe forward avec normalisation de couche et projection linéaire.
     * 
     * @param input Entrée de forme [seqLength, dModel]
     * @return Sortie projetée de forme [seqLength, outputSize]
     */
    public INDArray forward(INDArray input) {
        // Calcul de la moyenne et de la variance sur la dimension dModel (axis=1)
        INDArray mean = input.mean(1).reshape(input.rows(), 1); // [seqLength, 1]
        INDArray variance = input.var(false, 1).reshape(input.rows(), 1); // [seqLength, 1]
        INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [seqLength, 1]
        
        // Normalisation: (input - mean) / sqrt(variance + epsilon)
        INDArray normalized = input.sub(mean).mul(stdInv); // [seqLength, dModel]
        
        // Mise à l'échelle et décalage: normalized * gamma + beta
        INDArray scaled = normalized.mul(gamma).add(beta); // [seqLength, dModel]
        
        // Projection linéaire
        INDArray output = scaled.mmul(weights).addRowVector(bias); // [seqLength, outputSize]
        return output;
    }

    /**
     * Passe backward pour calculer les gradients.
     * 
     * @param input      Entrée originale utilisée dans la passe forward de forme [seqLength, dModel]
     * @param gradOutput Gradient provenant de la couche suivante de forme [seqLength, outputSize]
     * @return Map contenant les gradients pour les paramètres 'weights', 'bias', 'gamma' et 'beta', ainsi que 'input'
     */
    public Map<String, INDArray> backward(INDArray input, INDArray gradOutput) {
        // Calcul des moyennes et variances
        INDArray mean = input.mean(1).reshape(input.rows(), 1); // [seqLength, 1]
        INDArray variance = input.var(false, 1).reshape(input.rows(), 1); // [seqLength, 1]
        INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [seqLength, 1]

        // Calcul de normalized = (input - mean) / sqrt(var + epsilon)
        INDArray normalized = input.sub(mean).mul(stdInv); // [seqLength, dModel]

        // Gradients pour la projection linéaire
        INDArray gradScaled = gradOutput.mmul(weights.transpose()); // [seqLength, dModel]

        // Gradients pour gamma et beta de LayerNorm
        INDArray gradGamma = normalized.mul(gradScaled).sum(0); // [dModel]
        INDArray gradBeta = gradScaled.sum(0); // [dModel]

        // Gradients pour la normalisation
        INDArray gradNormalized = gradScaled.mul(gamma); // [seqLength, dModel]
        
        // Calcul des gradients pour l'entrée
        INDArray sumGradNormInput = gradNormalized.mul(normalized).sum(1).reshape(input.rows(), 1); // [seqLength, 1]
        INDArray gradInput = gradNormalized.mul(stdInv).sub(normalized.mul(sumGradNormInput).mul(stdInv)); // [seqLength, dModel]

        // Gradients pour les poids et les biais
        INDArray gradWeights = input.transpose().mmul(gradOutput); // [dModel, outputSize]
        INDArray gradBias = gradOutput.sum(0); // [outputSize]

        // Stockage des gradients dans la map
        gradients.put("weights", gradWeights);
        gradients.put("bias", gradBias);
        gradients.put("gamma", gradGamma);
        gradients.put("beta", gradBeta);
        gradients.put("input", gradInput); // Gradient à propager vers les couches précédentes

        return gradients;
    }

    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [weights, bias, gamma, beta]
     */
    public List<INDArray> getGradients() {
        return Arrays.asList(gradients.get("weights"), gradients.get("bias"), gradients.get("gamma"), gradients.get("beta"));
    }

    /**
     * Obtient les paramètres de la projection linéaire.
     * 
     * @return Liste des paramètres dans l'ordre [weights, bias, gamma, beta]
     */
    public List<INDArray> getParameters() {
        return Arrays.asList(weights, bias, gamma, beta);
    }

    /**
     * Définit (met à jour) les paramètres de la projection linéaire.
     * 
     * @param newWeights Nouvelles valeurs pour les poids
     * @param newBias    Nouvelles valeurs pour les biais
     * @param newGamma   Nouvelles valeurs pour gamma
     * @param newBeta    Nouvelles valeurs pour beta
     */
    public void setParameters(INDArray newWeights, INDArray newBias, INDArray newGamma, INDArray newBeta) {
        this.weights = newWeights;
        this.bias = newBias;
        this.gamma = newGamma;
        this.beta = newBeta;
    }

    /**
     * Obtient le nombre total de paramètres.
     * 
     * @return Nombre total de paramètres
     */
    public long getNumberOfParameters() {
        return weights.length() + bias.length() + gamma.length() + beta.length();
    }

    /**
     * Obtient le nombre total de gradients.
     * 
     * @return Nombre total de gradients
     */
    public long getNumberOfGradients() {
        return gradients.get("weights").length() + gradients.get("bias").length() + gradients.get("gamma").length() + gradients.get("beta").length();
    }
}
