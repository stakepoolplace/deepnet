package RN.utils;

import java.util.Arrays;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.transformer.Batch;
import RN.transformer.Tokenizer;

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
        INDArray safeInput = replaceInfinityWithLargeValue(input, 1e6, -1e6);
 
 
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
     * Remplace les valeurs infinies dans l'INDArray d'entrée par des valeurs finies.
     * Les valeurs +∞ sont remplacées par largeValuePos et -∞ par largeValueNeg.
     *
     * @param input          L'INDArray d'entrée.
     * @param largeValuePos  La grande valeur pour remplacer +∞ (ex: 1e6).
     * @param largeValueNeg  La grande valeur pour remplacer -∞ (ex: -1e6).
     * @return               L'INDArray modifié avec les infinies remplacées.
     */
    private static INDArray replaceInfinityWithLargeValue(INDArray input, double largeValuePos, double largeValueNeg) {
        INDArray result = input.dup(); // Dupliquer pour éviter de modifier l'original
        for (int i = 0; i < result.length(); i++) {
            double val = result.getDouble(i);
            if (Double.isInfinite(val)) {
                if (val > 0) {
                    result.putScalar(i, largeValuePos);
                } else {
                    result.putScalar(i, largeValueNeg);
                }
            }
        }
        return result;
    }


    public static void addGradient(Map<String, INDArray> gradients, String key, INDArray value) {
        if(value == null)
            throw new IllegalArgumentException("Tentative d'ajout de gradient null key : " + key);
        gradients.put(key, value);
    }

    public static INDArray stableSoftmax(INDArray logits, int axis) {
        INDArray max = logits.max(true, axis);
        INDArray shifted = logits.sub(max);
        INDArray exp = Transforms.exp(shifted);
        INDArray sumExp = exp.sum(true, axis);
        return exp.div(sumExp);
    }

   /**
     * Crée un masque de padding pour un batch donné.
     *
     * @param data INDArray contenant les IDs de tokens du batch [batchSize, seqLength].
     * @return INDArray représentant le masque de padding [batchSize, 1, 1, seqLength].
     */
    public static INDArray createKeyPaddingMask(Tokenizer tokenizer, INDArray tokens) {
        // tokens : [batchSize, seqLength]
        // output mask : [batchSize, 1, 1, seqLength]

    
        // Identifier les positions de padding
        INDArray paddingPositions = tokens.eq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT); // [batchSize, seqLength]
        // System.out.println("Padding Positions:\n" + paddingPositions);
    
        // Créer un masque initial rempli de 1.0f
        INDArray paddingMask = Nd4j.ones(DataType.FLOAT, tokens.size(0), 1, 1, tokens.size(1)); // [batchSize, 1, 1, seqLength]
    
        // Utiliser les opérations vectorisées pour soustraire les positions de padding
        INDArray paddingMaskFlat = paddingMask.reshape(tokens.size(0), tokens.size(1)); // [batchSize, seqLength]
        paddingMaskFlat = paddingMaskFlat.sub(paddingPositions); // 1 - 1 = 0 pour padding, 1 - 0 = 1 sinon
        paddingMask = paddingMaskFlat.reshape(tokens.size(0), 1, 1, tokens.size(1)); // [batchSize, 1, 1, seqLength]
    
        // System.out.println("Generated Padding Mask:\n" + paddingMask);
        return paddingMask;
    }

    
    public static INDArray createQueryPaddingMask(Tokenizer tokenizer, INDArray tokens) {
        // tokens : [batchSize, seqLength_q]
        // output mask : [batchSize, 1, seqLength_q, 1]
    
        // Identifier les positions de padding
        INDArray paddingPositions = tokens.eq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT); // [batchSize, seqLength_q]
    
        // Créer un masque où 1 indique un token valide et 0 indique un padding
        // Nous avons besoin d'un masque de forme [batchSize, 1, seqLength_q, 1] pour le broadcasting
        INDArray paddingMask = paddingPositions.reshape(tokens.size(0), 1, tokens.size(1), 1); // [batchSize, 1, seqLength_q, 1]
    
        // Inverser le masque pour que 1 indique un token valide et 0 indique un padding
        INDArray invertedMask = paddingMask.eq(0.0f).castTo(DataType.FLOAT); // 1.0f pour les tokens valides, 0.0f pour le padding
    
        // System.out.println("Generated Query Padding Mask:\n" + invertedMask);
        return invertedMask;
    }



    /**
     * Crée un masque look-ahead binaire pour le décodeur.
     *
     * @param batchSize Taille du batch.
     * @param size      Taille de la séquence.
     * @return INDArray représentant le masque look-ahead binaire [batchSize, 1, size, size].
     */
    public static INDArray createLookAheadMask(Integer batchSize, Integer size) {

       
        // Créer une matrice triangulaire inférieure remplie de 1.0f
        INDArray lookAheadMask = Nd4j.ones(DataType.FLOAT, size, size); // [size, size]
        
        // Remplir le masque avec 1.0f où j <= i et 0.0f où j > i
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (j > i) {
                    lookAheadMask.putScalar(new int[]{i, j}, 0.0f);
                } else {
                    lookAheadMask.putScalar(new int[]{i, j}, 1.0f);
                }
            }
        }
        
        // Reshaper pour correspondre aux dimensions attendues [1, 1, size, size]
        lookAheadMask = lookAheadMask.reshape(1, 1, size, size); // [1, 1, size, size]
        
        // Répéter le masque pour chaque exemple du batch
        INDArray repeatedMask = Nd4j.tile(lookAheadMask, batchSize, 1, 1, 1); // [batchSize, 1, size, size]
        
        return repeatedMask;
    }



}
