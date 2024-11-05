package RN.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NDArrayUtils {

    // Applique softmax sur une dimension spécifique de l'INDArray
    public static INDArray softmax(INDArray array, int softmaxDim) {
        // Si softmaxDim est -1, on le remplace par la dernière dimension
        if (softmaxDim == -1) {
            softmaxDim = array.rank() - 1;
        }

        // Si la dimension est déjà la dernière, on peut directement appliquer softmax
        if (softmaxDim == array.rank() - 1) {
            return Transforms.softmax(array, false);
        }
        
        // Sinon, on permute les dimensions pour mettre la dimension softmaxDim à la fin
        int[] permuteOrder = new int[array.rank()];
        for (int i = 0; i < array.rank(); i++) {
            permuteOrder[i] = i;
        }
        permuteOrder[softmaxDim] = array.rank() - 1;
        permuteOrder[array.rank() - 1] = softmaxDim;

        // Permuter pour mettre la dimension à la fin, appliquer softmax, puis remettre dans l'ordre d'origine
        INDArray permuted = array.permute(permuteOrder);
        INDArray softmaxResult = Transforms.softmax(permuted, false);

        // Re-permuter pour revenir à l'ordre d'origine
        return softmaxResult.permute(permuteOrder);
    }
}
