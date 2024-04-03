package RN.transformer;

import java.util.Arrays;
import java.util.List;

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

    // Méthode pour obtenir les paramètres (poids) de la projection
    public List<INDArray> getParameters() {
        return Arrays.asList(weights);
    }

    // Méthode pour définir (mettre à jour) les paramètres (poids) de la projection
    public void setParameters(INDArray newWeights) {
        this.weights = newWeights;
    }
}
