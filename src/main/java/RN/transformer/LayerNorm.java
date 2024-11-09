package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.utils.NDArrayUtils;

/**
 * Classe représentant une normalisation de couche (LayerNorm).
 * Les tenseurs peuvent avoir n'importe quel rang >= 2, normalisés sur la dernière dimension (dModel).
 */
public class LayerNorm extends Layer implements Serializable {
    private static final long serialVersionUID = 941772045774041840L;
    private INDArray gamma, beta;
    private int dModel;
    private final double epsilon = 1e-6;
    private INDArray inputCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();

    /**
     * Constructeur de la classe LayerNorm.
     * 
     * @param dModel Dimension du modèle (dModel)
     */
    public LayerNorm(int dModel) {
        this.dModel = dModel;
        // Initialisation de gamma à des uns et de beta à des zéros avec la forme [1, 1, dModel]
        gamma = Nd4j.ones(DataType.FLOAT, 1, 1, dModel); // [1, 1, dModel]
        beta = Nd4j.zeros(DataType.FLOAT, 1, 1, dModel); // [1, 1, dModel]

        // Log des formes pour vérification
        // System.out.println("Initialized gamma shape: " + Arrays.toString(gamma.shape()));
        // System.out.println("Initialized beta shape: " + Arrays.toString(beta.shape()));
    }

    /**
     * Passe forward de la normalisation de couche.
     * 
     * @param x Entrée de forme [batchSize, seqLength, dModel] ou plus
     * @return Sortie normalisée de même forme
     */
    @Override
    public INDArray forward(INDArray x) {
        if (x.isNaN().any() || x.isInfinite().any()) {
            throw new RuntimeException("LayerNorm.forward received NaN or Infinite values in input.");
        }

        this.inputCache = x.dup();

        // Calcul de la moyenne et de la variance sur la dernière dimension (dModel)
        INDArray mean = x.mean(true, x.rank() - 1);     // [batchSize, seqLength, 1]
        INDArray variance = x.var(false, x.rank() - 1);  // [batchSize, seqLength, 1]

        // Ajout de epsilon et calcul de l'écart-type
        INDArray std = Transforms.sqrt(variance.add(epsilon)).reshape(x.shape()[0], x.shape()[1], 1); // [batchSize, seqLength, 1]

        if (mean.isNaN().any() || mean.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in mean calculation.");
        }
        if (std.isNaN().any() || std.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values encountered in standard deviation calculation.");
        }

        // Normalisation avec broadcast explicite
        INDArray normalized = x.sub(mean).div(std); // [batchSize, seqLength, dModel] / [batchSize, seqLength, 1]

        // Reshape gamma et beta pour le broadcasting correct
        INDArray gammaBroadcast = gamma.reshape(1, 1, dModel);  // [1, 1, dModel]
        INDArray betaBroadcast = beta.reshape(1, 1, dModel);    // [1, 1, dModel]

        // Mise à l'échelle et décalage
        INDArray output = normalized.mul(gammaBroadcast).add(betaBroadcast); // [batchSize, seqLength, dModel]

        if (output.isNaN().any() || output.isInfinite().any()) {
            throw new RuntimeException("NaN or Infinite values produced by LayerNorm normalization.");
        }

        return output;
    }

    /**
     * Passe backward de la normalisation de couche.
     * 
     * @param gradOutput Gradient provenant de la couche suivante de même forme que l'entrée
     * @return Map contenant les gradients pour les paramètres 'gamma', 'beta' et 'input'
     */
    @Override
    public Map<String, INDArray> backward(INDArray gradOutput) {

        if (gradOutput == null) {
            throw new IllegalArgumentException("gradOutput ne peut pas être null lors de la rétropropagation dans LayerNorm.");
        }

        // Vérifications de base
        if (gradOutput.shape()[gradOutput.rank() - 1] != dModel) {
            throw new IllegalStateException("La dernière dimension de gradOutput doit être égale à dModel.");
        }

        // Récupération des valeurs du forward
        INDArray input = this.inputCache; // [batchSize, seqLength, dModel]
        long batchSize = input.size(0);
        long seqLength = input.size(1);
        // Note: On suppose que les dimensions intermédiaires sont maintenues

        // Recalcul de la moyenne et de la variance comme dans le forward
        INDArray mean = input.mean(true, input.rank() - 1); // [batchSize, seqLength, 1]
        INDArray variance = input.var(false, input.rank() - 1).reshape(batchSize, seqLength, 1); // [batchSize, seqLength, 1]

        INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [batchSize, seqLength, 1]

        INDArray normalized = input.sub(mean).mul(stdInv); // [batchSize, seqLength, dModel]

        // Calcul des gradients pour gamma et beta
        INDArray gradGamma = gradOutput.mul(normalized).sum(new int[]{0, 1}); // [1, 1, dModel]
        INDArray gradBeta = gradOutput.sum(new int[]{0, 1}); // [1, 1, dModel]

        // Calcul du gradient par rapport à l'entrée
        INDArray gradNormalized = gradOutput.mul(gamma); // [batchSize, seqLength, dModel]

        INDArray gradVariance = gradNormalized.mul(normalized).mul(-0.5).div(Transforms.pow(variance.add(epsilon), 1.5)).sum(2).reshape(batchSize, seqLength, 1); // [batchSize, seqLength, 1]
        INDArray gradMean = gradNormalized.mul(-1).div(Transforms.sqrt(variance.add(epsilon))).sum(2).reshape(batchSize, seqLength, 1)
                            .add( gradVariance.mul(normalized.mul(-2)).sum(2).reshape(batchSize, seqLength, 1)); // [batchSize, seqLength, 1]

        INDArray gradInput = gradNormalized.div(Transforms.sqrt(variance.add(epsilon)))
                            .add( gradVariance.mul(normalized.mul(2)).div(dModel))
                            .add( gradMean.div(dModel)); // [batchSize, seqLength, dModel]

        // Stockage des gradients
        NDArrayUtils.addGradient(gradients,"gamma", gradGamma);
        NDArrayUtils.addGradient(gradients,"beta", gradBeta);
        NDArrayUtils.addGradient(gradients,"input", gradInput);// Gradient à propager vers les couches précédentes

        return gradients;
    }

    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [gamma, beta]
     */
    public List<INDArray> getGradients() {
        List<INDArray> list = new ArrayList<>();
        list.add(gradients.get("gamma"));
        list.add(gradients.get("beta"));
        if (list.contains(null)) {
            throw new IllegalArgumentException(" gradients contains null ");
        }
        return list;  
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
        // Vérifier que les nouvelles formes sont correctes
        if (!Arrays.equals(newGamma.shape(), gamma.shape())) {
            throw new IllegalArgumentException("newGamma a une forme incorrecte: " + Arrays.toString(newGamma.shape()));
        }
        if (!Arrays.equals(newBeta.shape(), beta.shape())) {
            throw new IllegalArgumentException("newBeta a une forme incorrecte: " + Arrays.toString(newBeta.shape()));
        }
        this.gamma = newGamma;
        this.beta = newBeta;
    }

    /**
     * Obtient le nombre total de paramètres.
     * 
     * @return Nombre total de paramètres
     */
    public long getNumberOfParameters() {
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
