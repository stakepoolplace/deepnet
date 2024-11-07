package RN.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NDArrayUtils {

    // Applique softmax sur une dimension spécifique de l'INDArray
    public static INDArray softmax(INDArray array, int softmaxDim) {
        // Vérification des entrées
        if (array.isNaN().any()) {
            throw new IllegalArgumentException("Input array contains NaN before softmax");
        }

        // Si softmaxDim est -1, on le remplace par la dernière dimension
        if (softmaxDim == -1) {
            softmaxDim = array.rank() - 1;
        }

        // Si la dimension est déjà la dernière, on peut directement appliquer softmax
        if (softmaxDim == array.rank() - 1) {
            return Transforms.softmax(array, false);
        }
    
        // Normalisation pour éviter l'explosion numérique
        INDArray maxValues = array.max(true, softmaxDim);
        INDArray shiftedValues = array.sub(maxValues);
        
        // Calcul de l'exponentielle
        INDArray expValues = Transforms.exp(shiftedValues);
        
        // Somme pour la normalisation
        INDArray sumExp = expValues.sum(true, softmaxDim);
        
        // Division pour obtenir les probabilités
        INDArray softmaxOutput = expValues.div(sumExp);
        
        // Vérification de la sortie
        if (softmaxOutput.isNaN().any() || softmaxOutput.isInfinite().any()) {
            throw new RuntimeException("Softmax produced NaN or Infinite values");
        }
        
        // Verification that the sum of probabilities is close to 1
        INDArray probSum = softmaxOutput.sum(softmaxDim);
        INDArray diff = probSum.sub(1);

        // Use Transforms.abs() to calculate absolute values of the difference
        if (Transforms.abs(diff).maxNumber().doubleValue() > 1e-6) {
            throw new RuntimeException("Softmax probabilities do not sum to 1");
        }
        
        return softmaxOutput;
    }






}
