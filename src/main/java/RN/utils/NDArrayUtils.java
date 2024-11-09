package RN.utils;

import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NDArrayUtils {

    // // Applique softmax sur une dimension spécifique de l'INDArray
    // public static INDArray softmax(INDArray array, int softmaxDim) {
    //     // Vérification des entrées
    //     if (array.isNaN().any()) {
    //         throw new IllegalArgumentException("Input array contains NaN before softmax");
    //     }

    //     // Si softmaxDim est -1, on le remplace par la dernière dimension
    //     if (softmaxDim == -1) {
    //         softmaxDim = array.rank() - 1;
    //     }

    //     // Si la dimension est déjà la dernière, on peut directement appliquer softmax
    //     if (softmaxDim == array.rank() - 1) {
    //         return Transforms.softmax(array, false);
    //     }
    
    //     // Normalisation pour éviter l'explosion numérique
    //     INDArray maxValues = array.max(true, softmaxDim);
    //     INDArray shiftedValues = array.sub(maxValues);
        
    //     // Calcul de l'exponentielle
    //     INDArray expValues = Transforms.exp(shiftedValues);
        
    //     // Somme pour la normalisation
    //     INDArray sumExp = expValues.sum(true, softmaxDim);
        
    //     // Division pour obtenir les probabilités
    //     INDArray softmaxOutput = expValues.div(sumExp);
        
    //     // Vérification de la sortie
    //     if (softmaxOutput.isNaN().any() || softmaxOutput.isInfinite().any()) {
    //         throw new RuntimeException("Softmax produced NaN or Infinite values");
    //     }
        
    //     // Verification that the sum of probabilities is close to 1
    //     INDArray probSum = softmaxOutput.sum(softmaxDim);
    //     INDArray diff = probSum.sub(1);

    //     // Use Transforms.abs() to calculate absolute values of the difference
    //     if (Transforms.abs(diff).maxNumber().doubleValue() > 1e-6) {
    //         throw new RuntimeException("Softmax probabilities do not sum to 1");
    //     }
        
    //     return softmaxOutput;
    // }

    /**
     * Applique la fonction softmax à un INDArray le long d'une dimension spécifiée.
     * Si softmaxDim = -1, applique le softmax sur la dernière dimension.
     *
     * @param input       L'INDArray d'entrée.
     * @param softmaxDim  La dimension le long de laquelle appliquer la fonction softmax. Utiliser -1 pour la dernière dimension.
     * @return            L'INDArray après application de la fonction softmax.
     */
    public static INDArray softmax(INDArray input, int softmaxDim) {
 
        // Replace infinity values with a large finite number (1e6)
        INDArray safeInput = replaceInfinityWithLargeValue(input, 1e6);
 
 
        // Si softmaxDim est -1, le définir sur la dernière dimension
        if (softmaxDim == -1) {
            softmaxDim = safeInput.rank() - 1;
        }


        // Trouver la valeur maximale le long de la dimension spécifiée pour la stabilité numérique
        INDArray maxAlongDim = safeInput.max(true, softmaxDim);

        // Soustraire la valeur maximale de l'entrée
        INDArray shiftedInput = safeInput.sub(maxAlongDim);

        // Appliquer la fonction exponentielle
        INDArray expShifted = Transforms.exp(shiftedInput);

        // Calculer la somme des exponentielles le long de la dimension spécifiée
        INDArray sumExp = expShifted.sum(true, softmaxDim);

        // Diviser les exponentielles par la somme pour obtenir les probabilités softmax
        INDArray softmaxOutput = expShifted.div(sumExp);

        // Vérification de la sortie
        if (softmaxOutput.isNaN().any() || softmaxOutput.isInfinite().any()) {
            System.out.println("input : " + input);
            throw new RuntimeException("Softmax produced NaN or Infinite values");
        }

        return softmaxOutput;

    }


    /**
     * Replaces all ∞ values in the input INDArray with a large finite number.
     *
     * @param input     The input INDArray.
     * @param largeValue The large value to replace ∞ with (e.g., 1e6).
     * @return          The modified INDArray with ∞ replaced by largeValue.
     */
    private static INDArray replaceInfinityWithLargeValue(INDArray input, double largeValue) {
        INDArray result = input.dup(); // Duplicate to avoid modifying the original array
        for (int i = 0; i < result.length(); i++) {
            if (Double.isInfinite(result.getDouble(i))) {
                result.putScalar(i, largeValue);
            }
        }
        return result;
    }


    public static void addGradient(Map<String, INDArray> gradients, String key, INDArray value) {
        if(value == null)
            throw new IllegalArgumentException("Tentative d'ajout de gradient null key : " + key);
        gradients.put(key, value);
    }



}
