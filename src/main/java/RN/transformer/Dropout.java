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
        this.dropoutRate = dropoutRate;
    }

    public INDArray forward(boolean isTraining, INDArray input) {
        if (!isTraining || dropoutRate == 0.0) {
            return input;
        }

        // Créer un masque aléatoire
        INDArray mask = Nd4j.rand(input.shape()).gt(dropoutRate);
        
        // Convertir le masque en float avant la division
        INDArray maskFloat = mask.castTo(DataType.FLOAT);
        
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

}
