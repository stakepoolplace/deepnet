package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout implements Serializable{

    private static final long serialVersionUID = 1L;

    private final double rate;
    public INDArray mask;

    /**
     * Constructeur pour Dropout.
     * @param rate Taux de dropout (probabilité de désactivation d'un neurone).
     */
    public Dropout(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Le taux de dropout doit être entre 0.0 et 1.0 (non inclus).");
        }
        this.rate = rate;
    }

    /**
     * Applique le dropout.
     * @param training Indique si le mode est entraînement.
     * @param input Entrée à laquelle appliquer le dropout.
     * @param fixedMask Masque prédéfini (optionnel, utilisé principalement pour les tests).
     * @return Sortie après application du dropout.
     */
    public INDArray apply(boolean training, INDArray input, INDArray fixedMask) {
        if (training) {
            if (fixedMask != null) {
                mask = fixedMask;
            } else {
                // Générer un masque binaire avec probabilité (1 - rate) d'être 1
                mask = Nd4j.rand(input.shape()).gt(rate).castTo(input.dataType());
            }
            // Appliquer le masque et mettre à l'échelle par division
            return mask.mul(input).div(1.0 - rate);
        } else {
            // En inférence, ne rien faire
            return input;
        }
    }

    /**
     * Applique le dropout sans masque fixe (utilisé par défaut).
     */
    public INDArray apply(boolean training, INDArray input) {
        return apply(training, input, null);
    }

    /**
     * Calcule les gradients pour le dropout.
     * @param gradOutput Gradient de sortie.
     * @return Gradient d'entrée après application du masque et de la mise à l'échelle.
     */
    public INDArray backward(INDArray gradOutput) {
        if (mask == null) {
            throw new IllegalStateException("Le masque n'a pas été initialisé. Appelez la méthode apply en mode entraînement avant de calculer le gradient.");
        }
        // Appliquer le masque et mettre à l'échelle par division
        return mask.mul(gradOutput).div(1.0 - rate);
    }
}
