package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class PositionwiseFeedForward {
    private INDArray W1, W2;

    public PositionwiseFeedForward(int dModel, int dff) {
        W1 = Nd4j.rand(dModel, dff);
        W2 = Nd4j.rand(dff, dModel);
    }

    public INDArray forward(INDArray x) {
        return Transforms.relu(x.mmul(W1)).mmul(W2);
    }

    public List<INDArray> getParameters() {
        // Retourner les matrices de poids comme une liste d'INDArray
        return Arrays.asList(W1, W2);
    }
}
