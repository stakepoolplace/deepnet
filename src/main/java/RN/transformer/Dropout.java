package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout implements Serializable {
    private static final long serialVersionUID = -61325399079678110L;
    private double rate;
    INDArray mask;

    public Dropout(double rate) {
        if(rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Le taux de dropout doit être entre 0.0 et 1.0 (exclus).");
        }
        this.rate = rate;
    }

    /**
     * Applique le dropout à l'entrée.
     *
     * @param isTraining Indique si le modèle est en phase d'entraînement.
     * @param input      INDArray d'entrée [batchSize, ...].
     * @return INDArray après application du dropout.
     */
    public INDArray apply(boolean isTraining, INDArray input) {
        if(isTraining) {
            // Génération du masque où chaque élément a une probabilité (1 - rate) d'être activé
            this.mask = Nd4j.rand(input.shape()).gt(rate).castTo(input.dataType());
            // Application du masque et scaling (Inverted Dropout)
            return input.mul(mask).div(1.0 - rate);
        } else {
            // Pas de dropout pendant l'inférence
            return input;
        }
    }

    /**
     * Applique le masque de dropout aux gradients pendant la rétropropagation.
     *
     * @param gradOutput Gradients de sortie [batchSize, ...].
     * @return Gradients après application du masque de dropout.
     */
    public INDArray backward(INDArray gradOutput) {
        if(mask == null) {
            throw new IllegalStateException("Le masque doit être appliqué avant la rétropropagation.");
        }
        // Applique le masque et scaling pendant la rétropropagation
        return gradOutput.mul(mask).div(1.0 - rate);
    }
}
