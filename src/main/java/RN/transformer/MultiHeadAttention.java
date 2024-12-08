package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.utils.NDArrayUtils;

public class MultiHeadAttention implements Serializable {

    private static final long serialVersionUID = -7153801764592720027L;
    private int dModel;
    private int numHeads;
    private int depth;
    private INDArray inputQ, inputK, inputV; // Cached inputs for backward
    private INDArray Q, K, V; // Intermediate projections
    private INDArray Wq, Wk, Wv, Wo; // Weights for queries, keys, values, and output
    private INDArray gradInputQ, gradInputK, gradInputV;
    private INDArray attentionWeights; // Cached attention weights for backward
    private INDArray attentionOutput; // [batchSize * seqLength, numHeads * depth]
    private Map<String, INDArray> gradients = new HashMap<>();
    private INDArray lastAttentionScores;

    public MultiHeadAttention(int dModel, int numHeads) {

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.depth = dModel / numHeads;

        if (dModel != numHeads * depth) {
            throw new IllegalArgumentException("dModel doit être égal à numHeads * depth. Actuellement, dModel="
                    + dModel + ", numHeads=" + numHeads + ", depth=" + depth);
        }

        initializeWeights();

    }

    public void initializeWeights() {
        // Initialize weights with appropriate normalization
        // Initialisation des poids Wq, Wk, Wv, Wo avec Xavier Initialization
        this.Wq = Nd4j.randn(DataType.FLOAT, dModel, dModel).mul(1.0 / Math.sqrt(dModel)); // [dModel, dModel]
        this.Wk = Nd4j.randn(DataType.FLOAT, dModel, dModel).mul(1.0 / Math.sqrt(dModel)); // [dModel, dModel]
        this.Wv = Nd4j.randn(DataType.FLOAT, dModel, dModel).mul(1.0 / Math.sqrt(dModel)); // [dModel, dModel]
        this.Wo = Nd4j.randn(DataType.FLOAT, dModel, dModel).mul(1.0 / Math.sqrt(dModel)); // [dModel, dModel]
    }

    public INDArray forward(INDArray query, INDArray key, INDArray value, INDArray queryMask, INDArray keyMask, INDArray lookAheadMask) {
        // Validation des dimensions
        validateInputDimensions(query, key, value);
        
        // Cache les entrées pour le backward pass
        this.inputQ = query;
        this.inputK = key;
        this.inputV = value;

        int batchSize = (int) query.shape()[0];
        int seqLength_q = (int) query.shape()[1];
        int seqLength_k = (int) key.shape()[1];

        // Reshape pour la multiplication matricielle
        INDArray query2D = query.reshape(batchSize * seqLength_q, dModel);
        INDArray key2D = key.reshape(batchSize * seqLength_k, dModel);
        INDArray value2D = value.reshape(batchSize * seqLength_k, dModel);

        // Application des transformations linéaires
        this.Q = query2D.mmul(Wq);
        this.K = key2D.mmul(Wk);
        this.V = value2D.mmul(Wv);

        // Reshaping pour multi-head attention
        this.Q = this.Q.reshape(batchSize, seqLength_q, numHeads, depth).permute(0, 2, 1, 3);
        this.K = this.K.reshape(batchSize, seqLength_k, numHeads, depth).permute(0, 2, 1, 3);
        this.V = this.V.reshape(batchSize, seqLength_k, numHeads, depth).permute(0, 2, 1, 3);

        // Calcul des scores d'attention
        INDArray scores = Nd4j.create(batchSize, numHeads, seqLength_q, seqLength_k);
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray Q_batch_head = Q.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray KTransposed_batch_head = K.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).transpose();
                INDArray score = Q_batch_head.mmul(KTransposed_batch_head);
                scores.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).assign(score);
            }
        }

        // Normalisation
        scores = scores.div(Math.sqrt(depth));

        // Créer un masque combiné initial
        INDArray combinedMask = Nd4j.ones(batchSize, numHeads, seqLength_q, seqLength_k);

        // Masque de padding plus strict
        if (keyMask != null) {
            // Reshape le masque pour le broadcast
            INDArray paddingMask = keyMask;
            if (keyMask.rank() == 3) {
                paddingMask = keyMask.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());
            }
            paddingMask = paddingMask.reshape(batchSize, 1, 1, seqLength_k)
                                    .broadcast(batchSize, numHeads, seqLength_q, seqLength_k);
            combinedMask = combinedMask.mul(paddingMask);
        }

        // Masque causal plus strict pour le décodeur
        if (lookAheadMask != null) {
            // Masque causal plus strict
            INDArray causalMask = Nd4j.zeros(seqLength_q, seqLength_k);
            for (int i = 0; i < seqLength_q; i++) {
                for (int j = 0; j <= i && j < seqLength_k; j++) {
                    causalMask.putScalar(new int[]{i, j}, 1);
                }
            }
            
            // Appliquer le masque avec une pénalité plus forte
            causalMask = causalMask.reshape(1, 1, seqLength_q, seqLength_k)
                                  .broadcast(batchSize, numHeads, seqLength_q, seqLength_k);
            combinedMask = combinedMask.mul(causalMask);
            scores = scores.mul(combinedMask)
                          .add(combinedMask.rsub(1).mul(-1e9));
        }

        // Appliquer le masque aux scores avec une grande valeur négative pour les positions masquées
        scores = scores.mul(combinedMask)
                      .add(combinedMask.rsub(1).mul(-1e9));

        // System.out.println("Scores avant masque:\n" + scores);
        // System.out.println("combinedMask:\n" + combinedMask);

        // Application du softmax
        this.attentionWeights = NDArrayUtils.softmax(scores, -1);

        // Calcul de l'attention
        INDArray attention = Nd4j.create(batchSize, numHeads, seqLength_q, depth);
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray weights_batch_head = this.attentionWeights.get(NDArrayIndex.point(b), NDArrayIndex.point(h), 
                                                                       NDArrayIndex.all(), NDArrayIndex.all());
                INDArray V_batch_head = V.get(NDArrayIndex.point(b), NDArrayIndex.point(h), 
                                            NDArrayIndex.all(), NDArrayIndex.all());
                INDArray attention_head = weights_batch_head.mmul(V_batch_head);
                attention.get(NDArrayIndex.point(b), NDArrayIndex.point(h), 
                             NDArrayIndex.all(), NDArrayIndex.all()).assign(attention_head);
            }
        }

        // Reshape et multiplication finale
        INDArray attentionConcat = attention.permute(0, 2, 1, 3)
                                          .reshape(batchSize, seqLength_q, numHeads * depth);

        // Multiplication par Wo et reshape
        INDArray output = attentionConcat.reshape(batchSize * seqLength_q, dModel)
                                        .mmul(Wo)
                                        .reshape(batchSize, seqLength_q, dModel);

        // Masque final
        if (keyMask != null) {
            // Créer un masque final qui préserve la première dimension
            INDArray finalMask = keyMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                               
            // Étendre le masque pour correspondre à la forme de sortie [batchSize, seqLength_q, dModel]
            INDArray expandedMask = Nd4j.zeros(batchSize, seqLength_q, dModel);
            for (int i = 0; i < seqLength_q; i++) {
                for (int j = 0; j < dModel; j++) {
                    expandedMask.get(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.point(j))
                               .assign(finalMask.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)));
                }
            }
            
            output = output.mul(expandedMask);
            
        }

        // Cache pour backward pass
        this.attentionOutput = attentionConcat;

        // System.out.println("MultiHeadAttention Forward Pass:");
        // System.out.println("Q: " + Q);
        // System.out.println("K: " + K);
        // System.out.println("V: " + V);
        // System.out.println("Scores: " + scores);
        // System.out.println("Attention Weights: " + attentionWeights);
        // System.out.println("Attention Output: " + attentionOutput);
        // System.out.println("Output after Wo: " + output);

        // Stocker les scores d'attention
        this.lastAttentionScores = scores;

        return output;
    }

    /**
     * Backward pass of multi-head attention.
     *
     * @param gradOutput Gradient of the loss with respect to the output [batchSize,
     *                   seqLength, dModel]
     * @return Map of gradients with respect to parameters and inputs
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        // Vrifications d'état
        if (this.attentionOutput == null || this.Q == null || this.K == null || this.V == null) {
            throw new IllegalStateException(
                    "Les variables nécessaires (attentionOutput, Q, K, V) ne sont pas initialisées. Assurez-vous d'avoir effectué une passe forward avant backward.");
        }

        if (this.inputQ == null || this.inputK == null || this.inputV == null) {
            throw new IllegalStateException(
                    "Les inputs (inputQ, inputK, inputV) sont null. Assurez-vous que la passe forward les a correctement initialisés.");
        }

        if (this.attentionWeights == null) {
            throw new IllegalStateException(
                    "attentionWeights est null. Assurez-vous que la passe forward a correctement initialisé attentionWeights.");
        }

        // Dimensions
        int batchSize = (int) gradOutput.shape()[0];
        int seqLength = (int) gradOutput.shape()[1];
        int numHeads = this.numHeads;
        int depth = this.depth;
        int dModel = this.dModel; // Assurez-vous que dModel = numHeads * depth

        // Logs de dimensions
        // System.out.println("Backward Pass:");
        // System.out.println("batchSize: " + batchSize);
        // System.out.println("seqLength: " + seqLength);
        // System.out.println("numHeads: " + numHeads);
        // System.out.println("depth: " + depth);
        // System.out.println("dModel: " + dModel);
        // System.out.println("gradOutput shape: " + gradOutput.shapeInfoToString());

        // Step 1: Compute gradients for Wo
        // attentionOutputConcat has shape [batchSize, seqLength, numHeads * depth]
        INDArray attentionOutputConcat = this.attentionOutput.permute(0, 2, 1) // [batchSize, numHeads * depth,
                                                                               // seqLength]
                .reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]

        // Reshape gradOutput to [batchSize * seqLength, dModel]
        INDArray gradOutputReshaped = gradOutput.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength,
                                                                                         // dModel]

        // Compute gradWo: [numHeads * depth, dModel]
        INDArray gradWo = attentionOutputConcat.transpose().mmul(gradOutputReshaped); // [numHeads * depth, dModel]
        NDArrayUtils.addGradient(gradients, "Wo", gradWo);

        // System.out.println("gradWo shape: " + gradWo.shapeInfoToString());

        // Step 2: Compute gradients for attentionOutputConcat
        // gradAttentionOutputConcat = gradOutputReshaped * Wo^T
        INDArray WoTransposed = this.Wo.transpose(); // [dModel, numHeads * depth]
        INDArray gradAttentionOutputConcatReshaped = gradOutputReshaped.mmul(WoTransposed); // [batchSize * seqLength,
                                                                                            // numHeads * depth]
        // Reshape back to [batchSize, seqLength, numHeads, depth]
        INDArray gradAttentionOutputConcatND = gradAttentionOutputConcatReshaped.reshape(batchSize, seqLength, numHeads,
                depth);
        // Permute to [batchSize, numHeads, seqLength, depth]
        gradAttentionOutputConcatND = gradAttentionOutputConcatND.permute(0, 2, 1, 3); // [batchSize, numHeads,
                                                                                       // seqLength, depth]
        NDArrayUtils.addGradient(gradients, "gradAttentionOutputConcat", gradAttentionOutputConcatND);
        // System.out.println("gradAttentionOutputConcat shape: " +
        // gradAttentionOutputConcatND.shapeInfoToString());

        // Step 3: Compute gradients for attentionWeights and V
        // attentionOutput = attentionWeights.mmul(V)
        // gradAttentionWeights = gradAttentionOutputConcat.mmul(V^T)
        // gradV = attentionWeights^T.mmul(gradAttentionOutputConcat)
        INDArray gradAttentionWeights = Nd4j.create(batchSize, numHeads, seqLength, seqLength); // [batchSize, numHeads,
                                                                                                // seqLength, seqLength]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, depth] from gradAttentionOutputConcatND
                INDArray gradAttentionOutputHead = gradAttentionOutputConcatND.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()); // [seqLength, depth]

                // Extract [depth, seqLength] from V
                INDArray VTransposedHead = this.V.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).transpose(); // [depth, seqLength]

                // Perform mmul: [seqLength, depth] mmul [depth, seqLength] = [seqLength,
                // seqLength]
                INDArray gradAttentionWeightsHead = gradAttentionOutputHead.mmul(VTransposedHead); // [seqLength,
                                                                                                                         // seqLength]

                // Assign to gradAttentionWeights
                gradAttentionWeights.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).assign(gradAttentionWeightsHead);
            }
        }
        NDArrayUtils.addGradient(gradients, "gradAttentionWeights", gradAttentionWeights);
        // System.out.println("gradAttentionWeights shape: " +
        // gradAttentionWeights.shapeInfoToString());

        // Compute gradV
        INDArray gradV = Nd4j.create(batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, seqLength] from attentionWeights^T
                INDArray attentionWeightsTransposedHead = this.attentionWeights.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).transpose(); // [seqLength, seqLength]

                // Extract [seqLength, depth] from gradAttentionOutputConcatND
                INDArray gradAttentionOutputHead = gradAttentionOutputConcatND.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength,
                // depth]
                INDArray gradVHead = attentionWeightsTransposedHead.mmul(gradAttentionOutputHead); // [seqLength,
                                                                                                                         // depth]

                // Assign to gradV
                gradV.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).assign(gradVHead); // [seqLength, depth]

            }
        }
        NDArrayUtils.addGradient(gradients, "gradV", gradV);
        // System.out.println("gradV shape: " + gradV.shapeInfoToString());

        // Step 4: Compute gradients through softmax
        // gradScores = softmaxGrad(attentionWeights, gradAttentionWeights)
        INDArray gradScores = softmaxGrad(this.attentionWeights, gradAttentionWeights); // [batchSize, numHeads,
                                                                                        // seqLength, seqLength]
        NDArrayUtils.addGradient(gradients, "gradScores", gradScores);
        // System.out.println("gradScores shape: " + gradScores.shapeInfoToString());

        // Step 5: Compute gradients for Q and K
        // scores = Q.mmul(K.transpose(0, 1, 3, 2)) / sqrt(depth)
        // gradQ = gradScores.mmul(K) / sqrt(depth)
        // gradK = gradScores^T.mmul(Q) / sqrt(depth)
        INDArray gradQ = Nd4j.create(batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, seqLength] from gradScores
                INDArray gradScoresHead = gradScores.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()); // [seqLength, seqLength]

                // Extract [seqLength, depth] from K
                INDArray KHead = this.K.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength,
                // depth]
                INDArray gradQHead = gradScoresHead.mmul(KHead).div(Math.sqrt(depth)); // [seqLength, depth]

                // Assign to gradQ
                gradQ.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).assign(gradQHead);
            }
        }
        NDArrayUtils.addGradient(gradients, "gradQ", gradQ);
        // System.out.println("gradQ shape: " + gradQ.shapeInfoToString());

        // Compute gradK
        INDArray gradK = Nd4j.create(batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, seqLength] from gradScores transpose
                INDArray gradScoresTransposedHead = gradScores.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).transpose(); // [seqLength, seqLength]

                // Extract [seqLength, depth] from Q
                INDArray QHead = this.Q.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength,
                // depth]
                INDArray gradKHead = gradScoresTransposedHead.mmul(QHead).div(Math.sqrt(depth)); // [seqLength, depth]

                // Assign to gradK
                gradK.get(
                        NDArrayIndex.point(b),
                        NDArrayIndex.point(h),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()).assign(gradKHead);
            }
        }
        NDArrayUtils.addGradient(gradients, "gradK", gradK);
        // System.out.println("gradK shape: " + gradK.shapeInfoToString());

        // Step 6: Reshape gradQ and gradK for Wq and Wk gradients
        // Reshape gradQ and gradK from [batchSize, numHeads, seqLength, depth] to
        // [batchSize * seqLength, numHeads * depth]
        INDArray gradQReshaped = gradQ.reshape(batchSize * seqLength, numHeads * depth); // [1, 300]
        INDArray gradKReshaped = gradK.reshape(batchSize * seqLength, numHeads * depth); // [1, 300]

        // Step 7: Compute gradients for Wq, Wk, Wv
        // gradWq = inputQ^T * gradQReshaped [dModel, batchSize * seqLength] mmul
        // [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputQReshaped = this.inputQ.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWq = inputQReshaped.transpose().mmul(gradQReshaped); // [dModel, numHeads * depth]
        NDArrayUtils.addGradient(gradients, "Wq", gradWq);
        // System.out.println("gradWq shape: " + gradWq.shapeInfoToString());

        // gradWk = inputK^T * gradKReshaped [dModel, batchSize * seqLength] mmul
        // [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputKReshaped = this.inputK.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWk = inputKReshaped.transpose().mmul(gradKReshaped); // [dModel, numHeads * depth]
        NDArrayUtils.addGradient(gradients, "Wk", gradWk);
        // System.out.println("gradWk shape: " + gradWk.shapeInfoToString());

        // gradWv = inputV^T * gradVReshaped [dModel, batchSize * seqLength] mmul
        // [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        // Reshape gradV from [batchSize, numHeads, depth, seqLength] to [batchSize *
        // seqLength, numHeads * depth]
        INDArray gradVReshaped = gradV.permute(0, 1, 3, 2).reshape(batchSize * seqLength, numHeads * depth); // [1, 300]
        INDArray inputVReshaped = this.inputV.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWv = inputVReshaped.transpose().mmul(gradVReshaped); // [dModel, numHeads * depth]
        NDArrayUtils.addGradient(gradients, "Wv", gradWv);
        // System.out.println("gradWv shape: " + gradWv.shapeInfoToString());

        // // Step 8: Compute gradients for the inputs (query, key, value)

        // // For gradInputQ
        INDArray gradQPermuted = gradQ.permute(0, 2, 1, 3); // [batchSize, seqLength, numHeads, depth]
        gradQReshaped = gradQPermuted.reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength,
                                                                                        // numHeads * depth]
        gradInputQ = gradQReshaped.mmul(this.Wq.transpose()); // [batchSize * seqLength, dModel]
        gradInputQ = gradInputQ.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        NDArrayUtils.addGradient(gradients, "gradInputQ", gradInputQ);
        // System.out.println("gradInputQ shape: " + gradInputQ.shapeInfoToString());

        // // For gradInputK
        INDArray gradKPermuted = gradK.permute(0, 2, 1, 3);
        gradKReshaped = gradKPermuted.reshape(batchSize * seqLength, numHeads * depth);
        gradInputK = gradKReshaped.mmul(this.Wk.transpose());
        gradInputK = gradInputK.reshape(batchSize, seqLength, dModel);
        NDArrayUtils.addGradient(gradients, "gradInputK", gradInputK);
        // System.out.println("gradInputK shape: " + gradInputK.shapeInfoToString());

        // // For gradInputV
        INDArray gradVPermuted = gradV.permute(0, 2, 3, 1); // Adjusted permutation for gradV
        gradVReshaped = gradVPermuted.reshape(batchSize * seqLength, numHeads * depth);
        gradInputV = gradVReshaped.mmul(this.Wv.transpose());
        gradInputV = gradInputV.reshape(batchSize, seqLength, dModel);
        NDArrayUtils.addGradient(gradients, "gradInputV", gradInputV);
        // System.out.println("gradInputV shape: " + gradInputV.shapeInfoToString());

        // Step 7: Compute gradInput as gradOutput * Wo^T
        // Reshape gradOutput to 2D
        INDArray gradOutputReshapedFinal = gradOutput.reshape(batchSize * seqLength, dModel); // [6, 300]

        // Perform mmul with Wo^T
        INDArray gradInputReshaped = gradOutputReshapedFinal.mmul(WoTransposed); // [6, 300]

        // Reshape back to 3D
        INDArray gradInputFinal = gradInputReshaped.reshape(batchSize, seqLength, dModel); // [1, 6, 300]

        NDArrayUtils.addGradient(gradients, "input", gradInputFinal);
        // System.out.println("gradInput shape: " +
        // Arrays.toString(gradInputFinal.shape()));

        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            String key = entry.getKey();
            INDArray grad = entry.getValue();
            if (grad.isNaN().any() || grad.isInfinite().any()) {
                throw new RuntimeException("Gradient " + key + " contient des valeurs NaN ou infinies.");
            }
        }

        // Retourner les gradients des inputs séparément
        return gradients;
    }

    /**
     * Compute the gradient of the softmax function.
     *
     * @param softmaxOutput The output of the softmax function [batchSize, numHeads,
     *                      seqLength, seqLength]
     * @param gradOutput    The gradient w.r.t. the softmax output [batchSize,
     *                      numHeads, seqLength, seqLength]
     * @return The gradient w.r.t. the scores before softmax [batchSize, numHeads,
     *         seqLength, seqLength]
     */
    INDArray softmaxGrad(INDArray softmaxOutput, INDArray gradOutput) {
        // Calculate the sum over the last axis
        INDArray sum = Nd4j.sum(softmaxOutput.mul(gradOutput), 3).reshape(
                softmaxOutput.shape()[0],
                softmaxOutput.shape()[1],
                softmaxOutput.shape()[2],
                1); // [batchSize, numHeads, seqLength, 1]

        // Calculate gradScores
        INDArray gradScores = softmaxOutput.mul(gradOutput).sub(softmaxOutput.mul(sum)); // [batchSize, numHeads,
                                                                                         // seqLength, seqLength]

        return gradScores;
    }

    public List<INDArray> getParameters() {
        // Return weight matrices as a list of INDArrays
        return Arrays.asList(Wq, Wk, Wv, Wo);
    }

    public List<INDArray> getGradients() {
        // Return gradients as a list of INDArrays
        List<INDArray> list = new ArrayList<>();
        list.add(gradients.get("Wq"));
        list.add(gradients.get("Wk"));
        list.add(gradients.get("Wv"));
        list.add(gradients.get("Wo"));
        // Vérification pour s'assurer qu'aucun gradient n'est null
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == null) {
                throw new IllegalArgumentException("Le gradient pour le paramètre " + i + " est null.");
            }
        }
        return list;
    }

    public long getNumberOfParameters() {
        return Wq.length() + Wk.length() + Wv.length() + Wo.length();
    }

    public long getNumberOfGradients() {
        return gradients.get("Wq").length() + gradients.get("Wk").length() + gradients.get("Wv").length()
                + gradients.get("Wo").length();
    }

    // Getters and Setters
    public int getdModel() {
        return dModel;
    }

    public void setdModel(int dModel) {
        this.dModel = dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public void setNumHeads(int numHeads) {
        this.numHeads = numHeads;
    }

    public INDArray getWq() {
        return Wq;
    }

    public void setWq(INDArray wq) {
        Wq = wq;
    }

    public INDArray getWk() {
        return Wk;
    }

    public void setWk(INDArray wk) {
        Wk = wk;
    }

    public INDArray getWv() {
        return Wv;
    }

    public void setWv(INDArray wv) {
        Wv = wv;
    }

    public INDArray getWo() {
        return Wo;
    }

    public void setWo(INDArray wo) {
        Wo = wo;
    }

    public INDArray getAttentionWeights() {
        return this.attentionWeights;
    }

    public INDArray getLastAttentionScores() {
        return lastAttentionScores;
    }

    public void printAttentionWeights(List<String> queryTokens, List<String> keyTokens, int sampleIndex, Map<Integer, String> idToTokenMap) {
        if (this.attentionWeights == null) {
            System.out.println("Les poids d'attention ne sont pas disponibles. Effectuez d'abord une passe forward.");
            return;
        }
    
        // Vérifier que l'index de l'échantillon est valide
        if (sampleIndex >= attentionWeights.size(0)) {
            System.err.println("Index d'échantillon invalide: " + sampleIndex);
            return;
        }
    
        int numHeads = (int) attentionWeights.size(1);
        int seqLength_q = (int) attentionWeights.size(2);
        int seqLength_k = (int) attentionWeights.size(3);
    
        // Vérifier que la taille des queryTokens et keyTokens correspond aux dimensions des scores
        if (queryTokens.size() != seqLength_q) {
            System.err.println("La taille des queryTokens ne correspond pas à seqLength_q.");
            System.err.println("queryTokens.size() = " + queryTokens.size() + ", seqLength_q = " + seqLength_q);
            return;
        }
    
        if (keyTokens.size() != seqLength_k) {
            System.err.println("La taille des keyTokens ne correspond pas à seqLength_k.");
            System.err.println("keyTokens.size() = " + keyTokens.size() + ", seqLength_k = " + seqLength_k);
            return;
        }
    
        // Itérer sur chaque tête d'attention
        for (int head = 0; head < numHeads; head++) {
            System.out.println("===== Tête " + (head + 1) + " =====");
    
            // Imprimer les en-têtes avec les noms des tokens clés
            System.out.print(String.format("%-15s", "Requête"));
            for (String keyToken : keyTokens) {
                System.out.print(String.format("%-15s", keyToken));
            }
            System.out.println();
    
            // Afficher une ligne séparatrice
            int totalWidth = 15 + 15 * keyTokens.size();
            for (int i = 0; i < totalWidth; i++) {
                System.out.print("-");
            }
            System.out.println();
    
            // Itérer sur chaque token de la séquence de requêtes
            for (int q = 0; q < seqLength_q; q++) {
                // Récupérer le nom du token de requête
                String queryToken = (q < queryTokens.size()) ? queryTokens.get(q) : "Inconnu";
                System.out.print(String.format("%-15s", queryToken));
    
                for (int k = 0; k < seqLength_k; k++) {
                    // Récupérer le nom du token clé
                    String keyToken = (k < keyTokens.size()) ? keyTokens.get(k) : "Inconnu";
    
                    // Récupérer le poids d'attention pour cette paire (query, key)
                    double weight = attentionWeights.getDouble(sampleIndex, head, q, k);
    
                    // Optionnel : Arrondir le poids pour une meilleure lisibilité
                    String weightStr = String.format("%.4f", weight);
    
                    System.out.print(String.format("%-15s", weightStr));
                }
                System.out.println();
            }
    
            System.out.println(); // Ligne vide entre les têtes
        }
    }

    private INDArray calculateAttentionScores(INDArray Q, INDArray K, INDArray mask) {
        // Utilisation de depth au lieu de dModel pour le scaling
        float scalingFactor = (float) (1.0 / Math.sqrt(depth));
        
        INDArray scores = Q.mmul(K.permute(0, 2, 1));
        scores = scores.mul(scalingFactor);
        
        if (mask != null) {
            scores = scores.add(mask.mul(-1e9f));
        }
        
        return scores;
    }

    private void validateInputDimensions(INDArray query, INDArray key, INDArray value) {
        if (query.rank() != 3 || key.rank() != 3 || value.rank() != 3) {
            throw new IllegalArgumentException("Les entrées doivent être de rang 3 [batchSize, seqLength, dModel]");
        }
        
        if (query.size(2) != dModel || key.size(2) != dModel || value.size(2) != dModel) {
            throw new IllegalArgumentException("La dernière dimension des entrées doit être égale à dModel=" + dModel);
        }
        
        if (key.size(1) != value.size(1)) {
            throw new IllegalArgumentException("Les longueurs de séquence de key et value doivent être égales");
        }
    }

    public void clearCache() {
        inputQ = null;
        inputK = null;
        inputV = null;
        Q = K = V = null;
        attentionWeights = null;
        attentionOutput = null;
        lastAttentionScores = null;
        gradients.clear();
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            clearCache();
        } finally {
            super.finalize();
        }
    }

}
