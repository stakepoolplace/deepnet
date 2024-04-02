package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LayerNorm {
    private INDArray gamma, beta;
    private double epsilon = 1e-6;

    public LayerNorm(int dModel) {
        gamma = Nd4j.ones(dModel);
        beta = Nd4j.zeros(dModel);
    }

    public INDArray forward(INDArray x) {
        INDArray mean = x.mean(1);
        INDArray std = x.std(1);
        std.addi(epsilon);
        return x.sub(mean).divi(std).muli(gamma).addi(beta);
    }

    public List<INDArray> getParameters() {
        // Retourner une liste contenant les paramètres gamma et beta
        return Arrays.asList(gamma, beta);
    }
}
