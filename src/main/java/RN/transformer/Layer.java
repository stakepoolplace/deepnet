package RN.transformer;

import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

class Layer {
    // Paramètres de la couche (poids, biais, etc.)
    INDArray weights;
    INDArray bias;

    // Méthode pour calculer la passe en avant
    public INDArray forward(INDArray input) {
        // Implémentation spécifique de la couche
        return null;
    }

    // Méthode pour calculer la rétropropagation
    public Map<String, INDArray> backward(INDArray incomingGradient) {
        // Calculer les gradients par rapport aux paramètres de la couche
        // et propager le gradient de la perte à la couche précédente
        return null;
    }
}
