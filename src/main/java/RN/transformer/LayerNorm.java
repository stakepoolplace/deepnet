package RN.transformer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Classe représentant une normalisation de couche (LayerNorm).
 * Les tenseurs sont supposés avoir la forme [seqLength, dModel].
 */
public class LayerNorm extends Layer implements Serializable {
    private static final long serialVersionUID = 941772045774041840L;
    private INDArray gamma, beta;
    private final double epsilon = 1e-6;
    private INDArray inputCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();

    /**
     * Constructeur de la classe LayerNorm.
     * 
     * @param dModel Dimension du modèle (dModel)
     */
    public LayerNorm(int dModel) {
        // Initialisation de gamma à des uns et de beta à des zéros avec la forme [1, dModel]
        gamma = Nd4j.ones(1, dModel); // [1, dModel]
        beta = Nd4j.zeros(1, dModel); // [1, dModel]
    }

    /**
     * Passe forward de la normalisation de couche.
     * 
     * @param x Entrée de forme [seqLength, dModel]
     * @return Sortie normalisée de forme [seqLength, dModel]
     */
    @Override
    public INDArray forward(INDArray x) {
        // Vérification des valeurs NaN ou Inf dans l'entrée
        if (x.isNaN().any() || x.isInfinite().any()) {
            throw new RuntimeException("LayerNorm.forward received NaN or Infinite values in input.");
        }

        // Stockage de l'entrée pour la rétropropagation
        this.inputCache = x.dup();

        // Calcul de la moyenne et de la variance sur la dimension dModel (axis=1)
        INDArray mean = x.mean(1).reshape(x.rows(), 1); // [seqLength, 1]
        INDArray variance = x.var(false, 1).reshape(x.rows(), 1); // [seqLength, 1]

        // Ajout de epsilon pour éviter la division par zéro
        INDArray std = Transforms.sqrt(variance.add(epsilon)); // [seqLength, 1]

        // Vérification des valeurs NaN ou Inf dans les résultats intermédiaires
        if (mean.isNaN().any() || mean.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in mean calculation.");
        }
        if (std.isNaN().any() || std.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in standard deviation calculation.");
        }

        // Normalisation : (x - mean) / std
        INDArray normalized = x.sub(mean).div(std); // [seqLength, dModel]

        // Mise à l'échelle et décalage : normalized * gamma + beta
        INDArray output = normalized.mulRowVector(gamma).addRowVector(beta); // [seqLength, dModel]

        // Vérification des valeurs NaN ou Inf dans la sortie
        if (output.isNaN().any() || output.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values produced by LayerNorm normalization.");
        }

        return output;
    }

    /**
     * Passe backward de la normalisation de couche.
     * 
     * @param gradOutput Gradient provenant de la couche suivante de forme [seqLength, dModel]
     * @return Map contenant les gradients pour les paramètres 'gamma', 'beta' et 'input'
     */
    @Override
    public Map<String, INDArray> backward(INDArray gradOutput) {
        INDArray input = this.inputCache; // [seqLength, dModel]
        long seqLength = input.shape()[0];
        long dModel = input.shape()[1];

        // Recalcul de la moyenne et de la variance comme dans le forward
        INDArray mean = input.mean(1).reshape(seqLength, 1); // [seqLength, 1]
        INDArray variance = input.var(false, 1).reshape(seqLength, 1); // [seqLength, 1]
        INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [seqLength, 1]
        INDArray normalized = input.sub(mean).mul(stdInv); // [seqLength, dModel]

        // Reshape de gamma pour le broadcasting : [1, dModel]
        INDArray gammaReshaped = gamma.reshape(1, dModel); // [1, dModel]

        // Calcul des gradients pour gamma et beta
        INDArray gradGamma = gradOutput.mul(normalized).sum(0); // [dModel]
        INDArray gradBeta = gradOutput.sum(0); // [dModel]

        // Calcul du gradient par rapport à la normalisation
        INDArray gradNormalized = gradOutput.mul(gammaReshaped); // [seqLength, dModel]

        // Calcul du gradient par rapport à l'entrée
        INDArray sumGradNorm = gradNormalized.mul(normalized).sum(1).reshape(seqLength, 1); // [seqLength, 1]
        INDArray gradInput = gradNormalized.mul(stdInv).sub(normalized.mul(sumGradNorm).mul(stdInv)); // [seqLength, dModel]

        // Stockage des gradients dans la map
        gradients.put("gamma", gradGamma);
        gradients.put("beta", gradBeta);
        gradients.put("input", gradInput); // Gradient à propager vers les couches précédentes

        return gradients;
    }

    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [gamma, beta]
     */
    public List<INDArray> getGradients() {
        return Arrays.asList(gradients.get("gamma"), gradients.get("beta"));
    }

    /**
     * Obtient les paramètres de la normalisation de couche.
     * 
     * @return Liste des paramètres dans l'ordre [gamma, beta]
     */
    public List<INDArray> getParameters() {
        return Arrays.asList(gamma, beta);
    }

    /**
     * Définit (met à jour) les paramètres de la normalisation de couche.
     * 
     * @param newGamma Nouvelles valeurs pour gamma
     * @param newBeta  Nouvelles valeurs pour beta
     */
    public void setParameters(INDArray newGamma, INDArray newBeta) {
        this.gamma = newGamma;
        this.beta = newBeta;
    }

    /**
     * Obtient le nombre total de paramètres.
     * 
     * @return Nombre total de paramètres
     */
    public long getNumberOfParameters() {
        // gamma et beta ont chacun une taille de dModel
        return gamma.length() + beta.length();
    }

    /**
     * Obtient le nombre total de gradients.
     * 
     * @return Nombre total de gradients
     */
    public long getNumberOfGradients() {
        return gradients.get("gamma").length() + gradients.get("beta").length();
    }
}
