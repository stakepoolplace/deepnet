package RN.transformer;

import java.io.Serializable;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class PositionalEncoding implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 7621854975948659411L;
	private final int dModel; // Dimensionnalité des embeddings
    
    public PositionalEncoding(int dModel) {
        this.dModel = dModel;
    }

    public INDArray getPositionalEncoding(long sequenceLength) {
        INDArray positions = Nd4j.arange(sequenceLength).reshape(sequenceLength, 1);
        INDArray i = Nd4j.arange(dModel).reshape(1, dModel);
    
        // Créer un tableau compatible pour l'opération
        INDArray base = Nd4j.valueArrayOf(new long[]{1, dModel}, 10000.0);
        INDArray angleRates = Transforms.exp(Transforms.log(base).muli(2.0 / dModel).muli(i.neg()));
    
        INDArray angles = positions.mmul(angleRates); // [sequenceLength, dModel]
    
        for (int pos = 0; pos < sequenceLength; pos++) {
            for (int j = 0; j < dModel; j++) {
                double angle = angles.getDouble(pos, j);
                if (Double.isNaN(angle) || Double.isInfinite(angle)) {
                    throw new RuntimeException("PositionalEncoding contient des valeurs NaN ou infinies à la position " + pos + ", dimension " + j);
                }
                if (j % 2 == 0) {
                    angles.putScalar(new int[]{pos, j}, Math.sin(angle));
                } else {
                    angles.putScalar(new int[]{pos, j}, Math.cos(angle));
                }
            }
        }
    
        return angles; // [sequenceLength, dModel]
    }
    
    


}


