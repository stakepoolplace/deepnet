package RN.transformer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
    private INDArray attentionWeights; // Cached attention weights for backward
    private INDArray attentionOutput; // [batchSize * seqLength, numHeads * depth]
    private Map<String, INDArray> gradients = new HashMap<>();

    public MultiHeadAttention(int dModel, int numHeads) {

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.depth = dModel / numHeads;

        if (dModel != numHeads * depth) {
            throw new IllegalArgumentException("dModel doit être égal à numHeads * depth. Actuellement, dModel="
                    + dModel + ", numHeads=" + numHeads + ", depth=" + depth);
        }

        // Initialize weights with appropriate normalization
        Wq = Nd4j.randn(DataType.FLOAT, dModel, numHeads * depth).div(Math.sqrt(dModel));
        Wk = Nd4j.randn(DataType.FLOAT, dModel, numHeads * depth).div(Math.sqrt(dModel));
        Wv = Nd4j.randn(DataType.FLOAT, dModel, numHeads * depth).div(Math.sqrt(dModel));
        Wo = Nd4j.randn(DataType.FLOAT, numHeads * depth, dModel).div(Math.sqrt(numHeads * depth));
    }

    /**
     * Forward pass of multi-head attention.
     *
     * @param query Input queries of shape [batchSize, seqLength, dModel]
     * @param key   Input keys of shape [batchSize, seqLength, dModel]
     * @param value Input values of shape [batchSize, seqLength, dModel]
     * @param mask  Padding mask of shape [batchSize, 1, 1, seqLength]
     * @return Output of shape [batchSize, seqLength, dModel]
     */
    public INDArray forward(INDArray query, INDArray key, INDArray value, INDArray mask) {

        // Stocker les inputs pour la passe backward
        this.inputQ = query.dup(); // [batchSize, seqLength, dModel]
        this.inputK = key.dup();
        this.inputV = value.dup();

        // Obtention des dimensions
        int batchSize = (int) query.size(0);
        int seqLength = (int) query.size(1);

        // Vérification des dimensions des entrées
        if (query.rank() != 3 || key.rank() != 3 || value.rank() != 3) {
            throw new IllegalArgumentException("Les entrées query, key et value doivent être de rang 3.");
        }

        // Reshaping des entrées de [batchSize, seqLength, dModel] à [batchSize *
        // seqLength, dModel]
        INDArray query2D = query.reshape(batchSize * seqLength, dModel);
        INDArray key2D = key.reshape(batchSize * seqLength, dModel);
        INDArray value2D = value.reshape(batchSize * seqLength, dModel);

        // Application des transformations linéaires
        Q = query2D.mmul(Wq); // [batchSize * seqLength, numHeads * depth]
        K = key2D.mmul(Wk); // [batchSize * seqLength, numHeads * depth]
        V = value2D.mmul(Wv); // [batchSize * seqLength, numHeads * depth]

        // Reshaping pour multi-head attention
        Q = Q.reshape(batchSize, seqLength, numHeads, depth); // [batchSize, seqLength, numHeads, depth]
        K = K.reshape(batchSize, seqLength, numHeads, depth);
        V = V.reshape(batchSize, seqLength, numHeads, depth);

        // Transpose pour [batchSize, numHeads, seqLength, depth]
        Q = Q.permute(0, 2, 1, 3);
        K = K.permute(0, 2, 1, 3);
        V = V.permute(0, 2, 1, 3);

        // Initialiser un tableau pour stocker les scores
        INDArray scores = Nd4j.create(batchSize, numHeads, seqLength, seqLength); // [batchSize, numHeads, seqLength,
                                                                                  // seqLength]

        // Itérer sur batchSize et numHeads pour effectuer mmul
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extraire les matrices [seqLength, depth] et [depth, seqLength]
                INDArray Q_batch_head = Q.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(),
                        NDArrayIndex.all());
                INDArray KTransposed_batch_head = K
                        .get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all())
                        .transpose();

                // Calculer les scores [seqLength, seqLength]
                INDArray score = Q_batch_head.mmul(KTransposed_batch_head).div(Math.sqrt(depth));

                // Stocker les scores en utilisant assign
                scores.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all())
                        .assign(score);
            }
        }

        // Appliquer le masque si fourni
        if (mask != null) {
            // Assurez-vous que le masque a les mêmes dimensions que scores
            // Généralement, mask a la forme [batchSize, 1, 1, seqLength]
            scores = scores.add(mask.mul(-1e9)); // Ajouter un grand nombre négatif aux positions masquées
        }

        // Application du softmax sur le dernier axe (axis=3)
        // NDArrayUtils.softmax(scores, 3) doit être une méthode qui applique softmax
        // correctement
        // Assurez-vous que cette méthode fonctionne comme attendu
        INDArray weights = NDArrayUtils.softmax(scores, 3); // [batchSize, numHeads, seqLength, seqLength]

        // **Stocker attentionWeights pour la passe backward**
        this.attentionWeights = weights;

        // Calcul de l'attention pondérée
        INDArray attention = Nd4j.create(batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength,
                                                                                 // depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extraire les matrices [seqLength, seqLength] et [seqLength, depth]
                INDArray weights_batch_head = weights.get(NDArrayIndex.point(b), NDArrayIndex.point(h),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray V_batch_head = V.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(),
                        NDArrayIndex.all());

                // Calculer l'attention pondérée [seqLength, depth]
                INDArray attention_head = weights_batch_head.mmul(V_batch_head);

                // Stocker l'attention en utilisant assign
                attention.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all())
                        .assign(attention_head);
            }
        }

        // Transposition et reshaping pour concaténer les têtes
        // [batchSize, numHeads, seqLength, depth] -> [batchSize, seqLength, numHeads *
        // depth]
        INDArray attentionConcat = attention.permute(0, 2, 1, 3).reshape(batchSize, seqLength, numHeads * depth); // [batchSize,
                                                                                                                  // seqLength,
                                                                                                                  // dModel]

        // Effectuer la multiplication matricielle avec Wo
        // Reshape attentionConcat de [batchSize, seqLength, dModel] à [batchSize *
        // seqLength, dModel]
        INDArray attention2D = attentionConcat.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength,
                                                                                       // dModel]
        INDArray output2D = attention2D.mmul(Wo); // [batchSize * seqLength, dModel]

        // Reshape le résultat en [batchSize, seqLength, dModel]
        INDArray output = output2D.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]

        // **Stocker l'attentionOutput pour la passe backward**
        this.attentionOutput = attentionConcat; // ou `output` selon ce qui est nécessaire pour backward

        return output;
    }


    /**
     * Backward pass of multi-head attention.
     *
     * @param gradOutput Gradient of the loss with respect to the output [batchSize, seqLength, dModel]
     * @return Map of gradients with respect to parameters and inputs
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        // Vérifications d'état
        if (this.attentionOutput == null || this.Q == null || this.K == null || this.V == null) {
            throw new IllegalStateException("Les variables nécessaires (attentionOutput, Q, K, V) ne sont pas initialisées. Assurez-vous d'avoir effectué une passe forward avant backward.");
        }

        if (this.inputQ == null || this.inputK == null || this.inputV == null) {
            throw new IllegalStateException("Les inputs (inputQ, inputK, inputV) sont null. Assurez-vous que la passe forward les a correctement initialisés.");
        }

        if (this.attentionWeights == null) {
            throw new IllegalStateException("attentionWeights est null. Assurez-vous que la passe forward a correctement initialisé attentionWeights.");
        }

        Map<String, INDArray> gradients = new HashMap<>();

        // Dimensions
        int batchSize = (int) gradOutput.shape()[0];
        int seqLength = (int) gradOutput.shape()[1];
        int numHeads = this.numHeads;
        int depth = this.depth;
        int dModel = this.dModel; // Assurez-vous que dModel = numHeads * depth

        // Logs de dimensions
        System.out.println("Backward Pass:");
        System.out.println("batchSize: " + batchSize);
        System.out.println("seqLength: " + seqLength);
        System.out.println("numHeads: " + numHeads);
        System.out.println("depth: " + depth);
        System.out.println("dModel: " + dModel);
        System.out.println("gradOutput shape: " + gradOutput.shapeInfoToString());

        // Step 1: Compute gradients for Wo
        // attentionOutputConcat has shape [batchSize, seqLength, numHeads * depth]
        INDArray attentionOutputConcat = this.attentionOutput.permute(0, 2, 1) // [batchSize, numHeads * depth, seqLength]
                                            .reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]

        // Reshape gradOutput to [batchSize * seqLength, dModel]
        INDArray gradOutputReshaped = gradOutput.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]

        // Compute gradWo: [numHeads * depth, dModel]
        INDArray gradWo = attentionOutputConcat.transpose().mmul(gradOutputReshaped); // [numHeads * depth, dModel]
        gradients.put("Wo", gradWo);
        System.out.println("gradWo shape: " + gradWo.shapeInfoToString());

        // Step 2: Compute gradients for attentionOutputConcat
        // gradAttentionOutputConcat = gradOutputReshaped * Wo^T
        INDArray WoTransposed = this.Wo.transpose(); // [dModel, numHeads * depth]
        INDArray gradAttentionOutputConcatReshaped = gradOutputReshaped.mmul(WoTransposed); // [batchSize * seqLength, numHeads * depth]
        // Reshape back to [batchSize, seqLength, numHeads, depth]
        INDArray gradAttentionOutputConcatND = gradAttentionOutputConcatReshaped.reshape(batchSize, seqLength, numHeads, depth);
        // Permute to [batchSize, numHeads, seqLength, depth]
        gradAttentionOutputConcatND = gradAttentionOutputConcatND.permute(0, 2, 1, 3); // [batchSize, numHeads, seqLength, depth]
        gradients.put("gradAttentionOutputConcat", gradAttentionOutputConcatND);
        System.out.println("gradAttentionOutputConcat shape: " + gradAttentionOutputConcatND.shapeInfoToString());

        // Step 3: Compute gradients for attentionWeights and V
        // attentionOutput = attentionWeights.mmul(V)
        // gradAttentionWeights = gradAttentionOutputConcat.mmul(V^T)
        // gradV = attentionWeights^T.mmul(gradAttentionOutputConcat)
        INDArray gradAttentionWeights = Nd4j.create(batchSize, numHeads, seqLength, seqLength); // [batchSize, numHeads, seqLength, seqLength]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, depth] from gradAttentionOutputConcatND
                INDArray gradAttentionOutputHead = gradAttentionOutputConcatND.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ); // [seqLength, depth]

                // Extract [depth, seqLength] from V
                INDArray VTransposedHead = this.V.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).transpose(); // [depth, seqLength]

                // Perform mmul: [seqLength, depth] mmul [depth, seqLength] = [seqLength, seqLength]
                INDArray gradAttentionWeightsHead = gradAttentionOutputHead.mmul(VTransposedHead).div(Math.sqrt(depth)); // [seqLength, seqLength]

                // Assign to gradAttentionWeights
                gradAttentionWeights.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).assign(gradAttentionWeightsHead);
            }
        }
        gradients.put("gradAttentionWeights", gradAttentionWeights);
        System.out.println("gradAttentionWeights shape: " + gradAttentionWeights.shapeInfoToString());

        // Compute gradV
        INDArray gradV = Nd4j.create(batchSize, numHeads, depth, seqLength); // [batchSize, numHeads, depth, seqLength]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, seqLength] from attentionWeights^T
                INDArray attentionWeightsTransposedHead = this.attentionWeights.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).transpose(); // [seqLength, seqLength]

                // Extract [seqLength, depth] from gradAttentionOutputConcatND
                INDArray gradAttentionOutputHead = gradAttentionOutputConcatND.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength, depth]
                INDArray gradVHead = attentionWeightsTransposedHead.mmul(gradAttentionOutputHead).div(Math.sqrt(depth)); // [seqLength, depth]

                // Assign to gradV
                gradV.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).assign(gradVHead.transpose()); // [depth, seqLength]
            }
        }
        gradients.put("gradV", gradV);
        System.out.println("gradV shape: " + gradV.shapeInfoToString());

        // Step 4: Compute gradients through softmax
        // gradScores = softmaxGrad(attentionWeights, gradAttentionWeights)
        INDArray gradScores = softmaxGrad(this.attentionWeights, gradAttentionWeights); // [batchSize, numHeads, seqLength, seqLength]
        gradients.put("gradScores", gradScores);
        System.out.println("gradScores shape: " + gradScores.shapeInfoToString());

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
                    NDArrayIndex.all()
                ); // [seqLength, seqLength]

                // Extract [seqLength, depth] from K
                INDArray KHead = this.K.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength, depth]
                INDArray gradQHead = gradScoresHead.mmul(KHead); // [seqLength, depth]

                // Assign to gradQ
                gradQ.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).assign(gradQHead);
            }
        }
        gradients.put("gradQ", gradQ);
        System.out.println("gradQ shape: " + gradQ.shapeInfoToString());

        // Compute gradK
        INDArray gradK = Nd4j.create(batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Extract [seqLength, seqLength] from gradScores transpose
                INDArray gradScoresTransposedHead = gradScores.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).transpose(); // [seqLength, seqLength]

                // Extract [seqLength, depth] from Q
                INDArray QHead = this.Q.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ); // [seqLength, depth]

                // Perform mmul: [seqLength, seqLength] mmul [seqLength, depth] = [seqLength, depth]
                INDArray gradKHead = gradScoresTransposedHead.mmul(QHead); // [seqLength, depth]

                // Assign to gradK
                gradK.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).assign(gradKHead);
            }
        }
        gradients.put("gradK", gradK);
        System.out.println("gradK shape: " + gradK.shapeInfoToString());

        // Step 6: Reshape gradQ and gradK for Wq and Wk gradients
        // Reshape gradQ and gradK from [batchSize, numHeads, seqLength, depth] to [batchSize * seqLength, numHeads * depth]
        INDArray gradQReshaped = gradQ.reshape(batchSize * seqLength, numHeads * depth); // [1, 300]
        INDArray gradKReshaped = gradK.reshape(batchSize * seqLength, numHeads * depth); // [1, 300]

        // Step 7: Compute gradients for Wq, Wk, Wv
        // gradWq = inputQ^T * gradQReshaped [dModel, batchSize * seqLength] mmul [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputQReshaped = this.inputQ.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWq = inputQReshaped.transpose().mmul(gradQReshaped); // [dModel, numHeads * depth]
        gradients.put("Wq", gradWq);
        System.out.println("gradWq shape: " + gradWq.shapeInfoToString());

        // gradWk = inputK^T * gradKReshaped [dModel, batchSize * seqLength] mmul [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputKReshaped = this.inputK.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWk = inputKReshaped.transpose().mmul(gradKReshaped); // [dModel, numHeads * depth]
        gradients.put("Wk", gradWk);
        System.out.println("gradWk shape: " + gradWk.shapeInfoToString());

        // gradWv = inputV^T * gradVReshaped [dModel, batchSize * seqLength] mmul [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        // Reshape gradV from [batchSize, numHeads, depth, seqLength] to [batchSize * seqLength, numHeads * depth]
        INDArray gradVReshaped = gradV.permute(0, 1, 3, 2).reshape(batchSize * seqLength, numHeads * depth); // [1, 300]
        INDArray inputVReshaped = this.inputV.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWv = inputVReshaped.transpose().mmul(gradVReshaped); // [dModel, numHeads * depth]
        gradients.put("Wv", gradWv);
        System.out.println("gradWv shape: " + gradWv.shapeInfoToString());

        // Step 8: Compute gradients for the inputs (query, key, value)

        // For gradInputQ
        INDArray gradQPermuted = gradQ.permute(0, 2, 1, 3); // [batchSize, seqLength, numHeads, depth]
        gradQReshaped = gradQPermuted.reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]
        INDArray gradInputQ = gradQReshaped.mmul(this.Wq.transpose()); // [batchSize * seqLength, dModel]
        gradInputQ = gradInputQ.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        gradients.put("gradInputQ", gradInputQ);
        System.out.println("gradInputQ shape: " + gradInputQ.shapeInfoToString());

        // For gradInputK
        INDArray gradKPermuted = gradK.permute(0, 2, 1, 3);
        gradKReshaped = gradKPermuted.reshape(batchSize * seqLength, numHeads * depth);
        INDArray gradInputK = gradKReshaped.mmul(this.Wk.transpose());
        gradInputK = gradInputK.reshape(batchSize, seqLength, dModel);
        gradients.put("gradInputK", gradInputK);
        System.out.println("gradInputK shape: " + gradInputK.shapeInfoToString());

        // For gradInputV
        INDArray gradVPermuted = gradV.permute(0, 2, 3, 1); // Adjusted permutation for gradV
        gradVReshaped = gradVPermuted.reshape(batchSize * seqLength, numHeads * depth);
        INDArray gradInputV = gradVReshaped.mmul(this.Wv.transpose());
        gradInputV = gradInputV.reshape(batchSize, seqLength, dModel);
        gradients.put("gradInputV", gradInputV);
        System.out.println("gradInputV shape: " + gradInputV.shapeInfoToString());

        // Concatenate gradients of inputs along the last dimension
        INDArray gradInput = Nd4j.concat(2, gradInputQ, gradInputK, gradInputV); // [batchSize, seqLength, 3 * dModel]
        gradients.put("input", gradInput);

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
    private INDArray softmaxGrad(INDArray softmaxOutput, INDArray gradOutput) {
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
        return Arrays.asList(gradients.get("Wq"), gradients.get("Wk"), gradients.get("Wv"), gradients.get("Wo"));
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
}
