package RN.transformer;

import java.io.Serializable;

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
        INDArray angleRates = Transforms.pow(i.divi(dModel).muli(-2).divi(2), 10000.0);

        INDArray angles = positions.mmul(angleRates);
        angles.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, dModel)).assign(Transforms.sin(angles.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, dModel))));
        angles.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, dModel)).assign(Transforms.cos(angles.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, dModel))));

        return angles;
    }


}


