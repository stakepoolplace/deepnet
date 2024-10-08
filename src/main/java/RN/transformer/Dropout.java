package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = -61325399079678110L;
	private double rate;
    private INDArray mask;


    public Dropout(double rate) {
        this.rate = rate;
    }

    
    public INDArray apply(boolean isTraining, INDArray input) {
        if(isTraining) {
            // Création du masque d'activation basée sur le taux de dropout
            this.mask = Nd4j.rand(input.shape()).gt(rate);
            // Application du masque aux données d'entrée
            return input.mul(mask);
        } else {
            // Pendant l'inférence, dropout n'est pas appliqué mais les activations sont ajustées
            return input.mul(1.0 - rate);
        }
    }
    
    public INDArray backward(INDArray gradOutput) {
        // Pendant la rétropropagation, simplement passer le gradient à travers le masque
        return gradOutput.mul(mask);
    }    
    
}
