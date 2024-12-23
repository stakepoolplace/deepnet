package RN.utils;

import java.util.Arrays;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.transformer.Batch;
import RN.transformer.Tokenizer;

public class NDArrayUtils {

    /**
     * Applique la fonction softmax à un INDArray le long d'une dimension spécifiée.
     * Si softmaxDim = -1, applique le softmax sur la dernière dimension.
     *
     * @param input       L'INDArray d'entrée.
     * @param softmaxDim  La dimension le long de laquelle appliquer la fonction softmax. Utiliser -1 pour la dernière dimension.
     * @return            L'INDArray après application de la fonction softmax.
     */
    public static INDArray softmax(INDArray input, int axis) {
        INDArray maxValues = input.max(true, axis);
        INDArray shiftedInput = input.sub(maxValues);
        
        INDArray expValues = Transforms.exp(shiftedInput);
        INDArray sumExp = expValues.sum(true, axis);
        return expValues.div(sumExp);
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
     * @param sequenceLength      Taille de la séquence.
     * @return INDArray représentant le masque look-ahead binaire [batchSize, 1, size, size].
     */
    public static INDArray createLookAheadMask(Integer batchSize, Integer sequenceLength) {

       
        // Créer une matrice triangulaire inférieure remplie de 1.0f
        INDArray lookAheadMask = Nd4j.ones(DataType.FLOAT, sequenceLength, sequenceLength); // [size, size]
        
        // Remplir le masque avec 1.0f où j <= i et 0.0f où j > i
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                if (j > i) {
                    lookAheadMask.putScalar(new int[]{i, j}, 0.0f);
                } else {
                    lookAheadMask.putScalar(new int[]{i, j}, 1.0f);
                }
            }
        }
        
        // Reshaper pour correspondre aux dimensions attendues [1, 1, size, size]
        lookAheadMask = lookAheadMask.reshape(1, 1, sequenceLength, sequenceLength); // [1, 1, size, size]
        
        // Répéter le masque pour chaque exemple du batch
        INDArray repeatedMask = Nd4j.tile(lookAheadMask, batchSize, 1, 1, 1); // [batchSize, 1, size, size]
        
        return repeatedMask;
    }



    public static INDArray createPaddingMask(Tokenizer tokenizer, INDArray input) {
        INDArray mask = Nd4j.ones(input.shape());
        int padTokenId = tokenizer.getPadTokenId();
        
        // Mettre à 0 les positions où se trouve le token de padding
        for (int i = 0; i < input.shape()[0]; i++) {
            for (int j = 0; j < input.shape()[1]; j++) {
                if (input.getInt(i, j) == padTokenId) {
                    mask.putScalar(new int[]{i, j}, 0);
                }
            }
        }
        return mask;
    }


    public static INDArray eye(int rows, int cols) {
        INDArray eye = Nd4j.zeros(rows, cols);
        int min = Math.min(rows, cols);
        for (int i = 0; i < min; i++) {
            eye.putScalar(new int[]{i, i}, 1.0);
        }
        return eye;
    }  

    public static INDArray eye(int size) {
        return eye(size, size);
    }

    public static INDArray createNeutralMask(int batchSize, int seqLength) {
        return Nd4j.ones(batchSize, seqLength);
    }

}
