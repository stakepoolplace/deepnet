package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout implements Serializable {

    private static final long serialVersionUID = 1L;

    private double dropoutRate;
    private INDArray dropoutMask;

    public Dropout(double dropoutRate) {
        setDropoutRate(dropoutRate);
    }

    public void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Le taux de dropout doit être compris entre 0 et 1 (exclus)");
        }
        this.dropoutRate = rate;
    }

    public INDArray forward(boolean training, INDArray input) {
        if (!training || dropoutRate == 0.0) {
            return input;
        }

        // Créer un masque aléatoire
        dropoutMask = Nd4j.rand('u',input.shape()).gt(dropoutRate)
                     .div(1.0 - dropoutRate);
        
        // Appliquer le masque à l'entrée
        return input.mul(dropoutMask);
    }

    public INDArray backward(INDArray gradOutput) {
        if (dropoutRate == 0.0) {
            return gradOutput;
        }

        return gradOutput.mul(dropoutMask);
    }


}
