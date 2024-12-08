package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

public class Dropout implements Serializable {

    private static final long serialVersionUID = 1L;

    private double dropoutRate;
    private INDArray lastMask;

    public Dropout(double dropoutRate) {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw new IllegalArgumentException("Le taux de dropout doit être compris entre 0 et 1 (exclus)");
        }
        this.dropoutRate = dropoutRate;
    }

    public INDArray forward(boolean isTraining, INDArray input) {
        return forward(isTraining, input, null);
    }

    public INDArray forward(boolean isTraining, INDArray input, INDArray mask) {
        if (!isTraining || dropoutRate == 0.0) {
            return input;
        }

        // Utiliser le masque fourni ou en créer un nouveau
        INDArray maskFloat;
        if (mask != null) {
            maskFloat = mask.castTo(DataType.FLOAT);
        } else {
            INDArray randomMask = Nd4j.rand(input.shape()).gt(dropoutRate);
            maskFloat = randomMask.castTo(DataType.FLOAT);
        }
        
        // Mettre à l'échelle pour maintenir la moyenne
        maskFloat = maskFloat.div(1.0 - dropoutRate);
        
        this.lastMask = maskFloat;
        return input.mul(maskFloat);
    }

    public INDArray backward(INDArray gradOutput) {
        if (lastMask == null) {
            return gradOutput;
        }
        return gradOutput.mul(lastMask);
    }

    public void setDropoutRate(double rate) {
        this.dropoutRate = rate;
    }

    public double getDropoutRate() {
        return dropoutRate;
    }

    public INDArray getLastMask() {
        return lastMask;
    }

    
}
