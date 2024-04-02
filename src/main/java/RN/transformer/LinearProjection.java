package RN.transformer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearProjection {
    private INDArray weights;

    public LinearProjection(int inputSize, int outputSize) {
        // Initialisation des poids avec une distribution normale centrée autour de zéro et une variance de 1/sqrt(inputSize)
        this.weights = Nd4j.randn(inputSize, outputSize).divi(Math.sqrt(inputSize));
    }

    public INDArray project(INDArray input) {
        // Projection linéaire en multipliant l'entrée par les poids
        return input.mmul(weights);
    }
}