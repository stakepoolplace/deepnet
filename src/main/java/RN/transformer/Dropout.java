package RN.transformer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dropout {
    private double rate;

    public Dropout(double rate) {
        this.rate = rate;
    }

    public INDArray apply(INDArray x) {
        INDArray mask = Nd4j.rand(x.shape()).gt(rate);
        return x.mul(mask);
    }
}
