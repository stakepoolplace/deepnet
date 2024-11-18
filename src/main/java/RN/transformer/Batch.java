// Batch.java
package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Batch {
    private INDArray data;
    private INDArray target;
    private INDArray mask;
    private Tokenizer tokenizer;


    public Batch(INDArray data, INDArray target, INDArray mask) {
        this.data = data;
        this.target = target;
        this.mask = mask;

    }

    public Batch(List<String> data, List<String> target, Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.data = this.tokenizer.tokensToINDArray(data); // Convertit List<String> en INDArray
        this.target = this.tokenizer.tokensToINDArray(target);
        
        // System.out.println("Data IDs in Batch: " + Arrays.toString(this.data.toIntVector()));
        // System.out.println("Target IDs in Batch: " + Arrays.toString(this.target.toIntVector()));

    }

    public INDArray getData() {
        return data;
    }

    public INDArray getTarget() {
        return target;
    }

    public INDArray getMask() {
        return mask;
    }

    public INDArray getInputs() {
        return data;
    }

    public INDArray getTargets() {
        return target;
    }
}
