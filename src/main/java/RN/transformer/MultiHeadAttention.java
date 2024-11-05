package RN.transformer;

import java.io.Serializable;
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
    private INDArray attentionWeights; // Cached attention weights for backward
    private INDArray attentionOutput; // [batchSize * seqLength, numHeads * depth]
    private Map<String, INDArray> gradients = new HashMap<>();

    public MultiHeadAttention(int dModel, int numHeads) {
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.depth = dModel / numHeads;
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
        // Cache inputs for backward
        this.inputQ = query.dup();
        this.inputK = key.dup();
        this.inputV = value.dup();

        // Determine batch size and sequence length
        int batchSize = (int) query.shape()[0];
        int seqLength = (int) query.shape()[1];

        // Linear projections
        Q = query.reshape(batchSize * seqLength, dModel).mmul(Wq) // [batchSize * seqLength, numHeads * depth]
                 .reshape(batchSize, seqLength, numHeads, depth) // [batchSize, seqLength, numHeads, depth]
                 .permute(0, 2, 1, 3); // [batchSize, numHeads, seqLength, depth]

        K = key.reshape(batchSize * seqLength, dModel).mmul(Wk) // [batchSize * seqLength, numHeads * depth]
               .reshape(batchSize, seqLength, numHeads, depth) // [batchSize, seqLength, numHeads, depth]
               .permute(0, 2, 1, 3); // [batchSize, numHeads, seqLength, depth]

        V = value.reshape(batchSize * seqLength, dModel).mmul(Wv) // [batchSize * seqLength, numHeads * depth]
                 .reshape(batchSize, seqLength, numHeads, depth) // [batchSize, seqLength, numHeads, depth]
                 .permute(0, 2, 1, 3); // [batchSize, numHeads, seqLength, depth]

        // Transpose K for attention score computation
        INDArray kTranspose = K.permute(0, 1, 3, 2); // [batchSize, numHeads, depth, seqLength]

        // Compute attention scores
        INDArray attentionScores = Nd4j.matmul(Q, kTranspose).div(Math.sqrt(depth)); // [batchSize, numHeads, seqLength, seqLength]

        // Apply mask if provided
        if (mask != null) {
            attentionScores.addi(mask); // Mask irrelevant scores
        }

        // Compute attention weights using softmax on the last dimension
        attentionWeights = NDArrayUtils.softmax(attentionScores, -1); // [batchSize, numHeads, seqLength, seqLength]

        // Compute attention output
        INDArray attentionOutputND = Nd4j.matmul(attentionWeights, V); // [batchSize, numHeads, seqLength, depth]

        // Permute and reshape to combine heads
        INDArray reshapedAttentionOutput = attentionOutputND.permute(0, 2, 1, 3) // [batchSize, seqLength, numHeads, depth]
                                                      .reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]
        this.attentionOutput = reshapedAttentionOutput; // Cache for backward

        // Final linear projection
        INDArray finalOutput = reshapedAttentionOutput.mmul(Wo).reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        System.out.println("Shape of Final Output: " + finalOutput.shapeInfoToString());

        return finalOutput; // [batchSize, seqLength, dModel]
    }

    /**
     * Backward pass of multi-head attention.
     *
     * @param gradOutput Gradient of the loss with respect to the output [batchSize, seqLength, dModel]
     * @return Gradients with respect to parameters and inputs
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        if (this.attentionOutput == null || this.Q == null || this.K == null || this.V == null) {
            throw new IllegalStateException("Necessary variables (attentionOutput, Q, K, V) are not initialized. Ensure forward pass is called before backward.");
        }

        Map<String, INDArray> gradients = new HashMap<>();

        // Dimensions
        int batchSize = (int) gradOutput.shape()[0];
        int seqLength = (int) gradOutput.shape()[1];
        int dModel = this.dModel;
        int numHeads = this.numHeads;
        int depth = this.depth;

        // Step 1: Reshape gradOutput from [batchSize, seqLength, dModel] to [batchSize * seqLength, dModel]
        INDArray gradOutputReshaped = gradOutput.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        System.out.println("gradOutputReshaped shape: " + gradOutputReshaped.shapeInfoToString());

        // Step 2: Compute gradWo = attentionOutput^T [numHeads * depth, batchSize * seqLength] mmul gradOutputReshaped [batchSize * seqLength, dModel] = [numHeads * depth, dModel]
        INDArray gradWo = this.attentionOutput.transpose().mmul(gradOutputReshaped); // [numHeads * depth, dModel]
        gradients.put("Wo", gradWo);
        System.out.println("Gradient Wo shape: " + gradWo.shapeInfoToString());

        // Step 3: Compute gradAttentionOutputReshaped = gradOutputReshaped [batchSize * seqLength, dModel] mmul Wo^T [dModel, numHeads * depth] = [batchSize * seqLength, numHeads * depth]
        INDArray gradAttentionOutputReshaped = gradOutputReshaped.mmul(Wo.transpose()); // [batchSize * seqLength, numHeads * depth]
        System.out.println("gradAttentionOutputReshaped shape: " + gradAttentionOutputReshaped.shapeInfoToString());

        // Step 4: Reshape gradAttentionOutputReshaped from [batchSize * seqLength, numHeads * depth] to [batchSize, seqLength, numHeads, depth]
        INDArray gradAttentionOutput = gradAttentionOutputReshaped.reshape(batchSize, seqLength, numHeads, depth); // [batchSize, seqLength, numHeads, depth]
        System.out.println("gradAttentionOutput shape: " + gradAttentionOutput.shapeInfoToString());

        // Step 5: Permute to [batchSize, numHeads, seqLength, depth]
        gradAttentionOutput = gradAttentionOutput.permute(0, 2, 1, 3); // [batchSize, numHeads, seqLength, depth]
        System.out.println("gradAttentionOutput permuted shape: " + gradAttentionOutput.shapeInfoToString());

        // Step 6: Compute gradV = attentionWeights [batchSize, numHeads, seqLength, seqLength] mmul gradAttentionOutput [batchSize, numHeads, seqLength, depth] = [batchSize, numHeads, seqLength, depth]
        INDArray gradV = Nd4j.create(DataType.FLOAT, batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray attentionWeightsHead = this.attentionWeights.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, seqLength); // [seqLength, seqLength]
                INDArray gradAttentionOutputHead = gradAttentionOutput.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, depth); // [seqLength, depth]
                INDArray gradVHead = attentionWeightsHead.mmul(gradAttentionOutputHead); // [seqLength, depth]
                // Use assign instead of put to assign the entire sub-array
                gradV.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).assign(gradVHead); // [b, h, :, :]
            }
        }
        System.out.println("gradV shape: " + gradV.shapeInfoToString());

        // Step 7: Reshape gradV from [batchSize, numHeads, seqLength, depth] to [batchSize * seqLength, numHeads * depth]
        INDArray gradVReshaped = gradV.permute(0, 2, 1, 3).reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]
        System.out.println("gradVReshaped shape: " + gradVReshaped.shapeInfoToString());

        // Step 8: Compute gradWv = inputV^T [dModel, batchSize * seqLength] mmul gradVReshaped [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputVReshaped = this.inputV.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWv = inputVReshaped.transpose().mmul(gradVReshaped); // [dModel, numHeads * depth]
        gradients.put("Wv", gradWv);
        System.out.println("Gradient Wv shape: " + gradWv.shapeInfoToString());

        // Step 9: Compute gradScores = gradAttentionOutput [batchSize, numHeads, seqLength, depth] mmul V^T [batchSize, numHeads, depth, seqLength] = [batchSize, numHeads, seqLength, seqLength]
        INDArray gradScores = Nd4j.create(DataType.FLOAT, batchSize, numHeads, seqLength, seqLength); // [batchSize, numHeads, seqLength, seqLength]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray gradAttentionOutputHead = gradAttentionOutput.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, depth); // [seqLength, depth]
                // Correctly transpose VHeadT to [depth, seqLength]
                INDArray VHeadT = this.V.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(depth, seqLength); // [depth, seqLength]
                INDArray gradScoresHead = gradAttentionOutputHead.mmul(VHeadT).div(Math.sqrt(depth)); // [seqLength, seqLength]
                // Use assign instead of put to assign the entire sub-array
                gradScores.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).assign(gradScoresHead); // [b, h, :, :]
            }
        }
        System.out.println("gradScores shape: " + gradScores.shapeInfoToString());

        // Step 10: Compute gradAttentionScoresFinal = softmaxGrad(attentionWeights, gradScores)
        INDArray gradAttentionScoresFinal = softmaxGrad(this.attentionWeights, gradScores); // [batchSize, numHeads, seqLength, seqLength]
        gradients.put("attentionScores", gradAttentionScoresFinal);
        System.out.println("gradAttentionScoresFinal shape: " + gradAttentionScoresFinal.shapeInfoToString());

        // Step 11: Compute gradQ = gradAttentionScoresFinal [batchSize, numHeads, seqLength, seqLength] mmul K [batchSize, numHeads, seqLength, depth] = [batchSize, numHeads, seqLength, depth]
        INDArray gradQ = Nd4j.create(DataType.FLOAT, batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray gradScoresHead = gradAttentionScoresFinal.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, seqLength); // [seqLength, seqLength]
                INDArray KHead = this.K.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, depth); // [seqLength, depth]
                INDArray gradQHead = gradScoresHead.mmul(KHead); // [seqLength, depth]
                // Use assign instead of put to assign the entire sub-array
                gradQ.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).assign(gradQHead); // [b, h, :, :]
            }
        }
        System.out.println("gradQ shape: " + gradQ.shapeInfoToString());

        // Step 12: Reshape gradQ from [batchSize, numHeads, seqLength, depth] to [batchSize * seqLength, numHeads * depth]
        INDArray gradQReshaped = gradQ.permute(0, 2, 1, 3).reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]
        System.out.println("gradQReshaped shape: " + gradQReshaped.shapeInfoToString());

        // Step 13: Compute gradWq = inputQ^T [dModel, batchSize * seqLength] mmul gradQReshaped [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputQReshaped = this.inputQ.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWq = inputQReshaped.transpose().mmul(gradQReshaped); // [dModel, numHeads * depth]
        gradients.put("Wq", gradWq);
        System.out.println("Gradient Wq shape: " + gradWq.shapeInfoToString());

        // Step 14: Compute gradK = gradAttentionScoresFinal [batchSize, numHeads, seqLength, seqLength] mmul Q [batchSize, numHeads, seqLength, depth] = [batchSize, numHeads, seqLength, depth]
        INDArray gradK = Nd4j.create(DataType.FLOAT, batchSize, numHeads, seqLength, depth); // [batchSize, numHeads, seqLength, depth]
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                INDArray gradScoresHeadTrans = gradAttentionScoresFinal.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, seqLength).transpose(); // [seqLength, seqLength]
                INDArray QHead = this.Q.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(h),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
                ).reshape(seqLength, depth); // [seqLength, depth]
                INDArray gradKHead = gradScoresHeadTrans.mmul(QHead); // [seqLength, depth]
                // Use assign instead of put to assign the entire sub-array
                gradK.get(NDArrayIndex.point(b), NDArrayIndex.point(h), NDArrayIndex.all(), NDArrayIndex.all()).assign(gradKHead); // [b, h, :, :]
            }
        }
        System.out.println("gradK shape: " + gradK.shapeInfoToString());

        // Step 15: Reshape gradK from [batchSize, numHeads, seqLength, depth] to [batchSize * seqLength, numHeads * depth]
        INDArray gradKReshaped = gradK.permute(0, 2, 1, 3).reshape(batchSize * seqLength, numHeads * depth); // [batchSize * seqLength, numHeads * depth]
        System.out.println("gradKReshaped shape: " + gradKReshaped.shapeInfoToString());

        // Step 16: Compute gradWk = inputK^T [dModel, batchSize * seqLength] mmul gradKReshaped [batchSize * seqLength, numHeads * depth] = [dModel, numHeads * depth]
        INDArray inputKReshaped = this.inputK.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        INDArray gradWk = inputKReshaped.transpose().mmul(gradKReshaped); // [dModel, numHeads * depth]
        gradients.put("Wk", gradWk);
        System.out.println("Gradient Wk shape: " + gradWk.shapeInfoToString());

        // Step 17: Compute gradInputQ = gradQReshaped [batchSize * seqLength, numHeads * depth] mmul Wq^T [numHeads * depth, dModel] = [batchSize * seqLength, dModel]
        INDArray gradInputQ = gradQReshaped.mmul(Wq.transpose()); // [batchSize * seqLength, dModel]
        gradInputQ = gradInputQ.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        gradients.put("inputQ", gradInputQ);
        System.out.println("gradInputQ shape: " + gradInputQ.shapeInfoToString());

        // Step 18: Compute gradInputK = gradKReshaped [batchSize * seqLength, numHeads * depth] mmul Wk^T [numHeads * depth, dModel] = [batchSize * seqLength, dModel]
        INDArray gradInputK = gradKReshaped.mmul(Wk.transpose()); // [batchSize * seqLength, dModel]
        gradInputK = gradInputK.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        gradients.put("inputK", gradInputK);
        System.out.println("gradInputK shape: " + gradInputK.shapeInfoToString());

        // Step 19: Compute gradInputV = gradVReshaped [batchSize * seqLength, numHeads * depth] mmul Wv^T [numHeads * depth, dModel] = [batchSize * seqLength, dModel]
        INDArray gradInputV = gradVReshaped.mmul(Wv.transpose()); // [batchSize * seqLength, dModel]
        gradInputV = gradInputV.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        gradients.put("inputV", gradInputV);
        System.out.println("gradInputV shape: " + gradInputV.shapeInfoToString());

        // Step 20: Compute gradInput = gradInputQ + gradInputK + gradInputV
        INDArray gradInput = gradInputQ.add(gradInputK).add(gradInputV); // [batchSize, seqLength, dModel]
        gradients.put("input", gradInput);
        System.out.println("gradInput shape: " + gradInput.shapeInfoToString());

        return gradients;
    }

    /**
     * Computes the gradient of the softmax.
     *
     * @param softmax Softmax results of shape [batchSize, numHeads, seqLength, seqLength]
     * @param gradA   Gradients from the next layer of the same shape as softmax
     * @return Gradient with respect to the attention scores of the same shape as softmax
     */
    private INDArray softmaxGrad(INDArray softmax, INDArray gradA) {
        // softmax: [batchSize, numHeads, seqLength, seqLength]
        // gradA: [batchSize, numHeads, seqLength, seqLength]

        // Compute dL/dS = softmax * (gradA - sum(gradA * softmax, axis=3, keepdims=true))
        // ND4J does not support keepdims directly, so handle it manually

        // Compute sum(gradA * softmax, axis=3, keepdims=true)
        INDArray sum = softmax.mul(gradA).sum(3).reshape(softmax.shape()[0], softmax.shape()[1], softmax.shape()[2], 1); // [batchSize, numHeads, seqLength, 1]

        // Compute gradS = softmax * (gradA - sum)
        INDArray gradS = softmax.mul(gradA.sub(sum)); // [batchSize, numHeads, seqLength, seqLength]

        return gradS;
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
