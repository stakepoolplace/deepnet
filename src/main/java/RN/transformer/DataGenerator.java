// DataGenerator.java
package RN.transformer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.stream.Collectors;

public class DataGenerator {
    private List<List<Integer>> inputSequences;
    private List<List<Integer>> targetSequences;
    private Tokenizer tokenizer;
    private int batchSize;
    private int sequenceLength;
    private int currentBatch;

    public DataGenerator(List<String> data, List<String> targets, Tokenizer tokenizer, int batchSize, int sequenceLength) {
        this.tokenizer = tokenizer;
        this.batchSize = batchSize;
        this.sequenceLength = sequenceLength;
        this.currentBatch = 0;
        this.inputSequences = data.stream()
                                  .map(this::preprocess)
                                  .collect(Collectors.toList());
        this.targetSequences = targets.stream()
                                      .map(this::preprocess)
                                      .collect(Collectors.toList());
    }

    private List<Integer> preprocess(String text) {
        List<String> tokens = tokenizer.tokenize(text);
        List<Integer> ids = tokenizer.tokensToIds(tokens);
        ids.add(0, tokenizer.getStartTokenId());
        ids.add(tokenizer.getEndTokenId());
        while (ids.size() < sequenceLength) {
            ids.add(tokenizer.getPadTokenId());
        }
        if (ids.size() > sequenceLength) {
            ids = ids.subList(0, sequenceLength);
        }
        return ids;
    }

    public boolean hasNextBatch() {
        return currentBatch * batchSize < inputSequences.size();
    }

    public Batch nextBatch() {
        return getNextBatch();
    }

    public Batch getNextBatch() {
        if (!hasNextBatch()) {
            return null; // Aucun batch restant
        }

        int start = currentBatch * batchSize;
        int end = Math.min(start + batchSize, inputSequences.size());

        List<List<Integer>> batchInputs = inputSequences.subList(start, end);
        List<List<Integer>> batchTargets = targetSequences.subList(start, end);

        int actualBatchSize = batchInputs.size();

        // Créer un INDArray de type entier pour les inputs
        INDArray inputs = Nd4j.create(DataType.INT32, actualBatchSize, sequenceLength);

        // Créer un INDArray de type entier pour les targets
        INDArray targets = Nd4j.create(DataType.INT32, actualBatchSize, sequenceLength);

        for (int i = 0; i < batchInputs.size(); i++) {
            for (int j = 0; j < batchInputs.get(i).size(); j++) {
                inputs.putScalar(new int[]{i, j}, batchInputs.get(i).get(j));
                targets.putScalar(new int[]{i, j}, batchTargets.get(i).get(j));
            }
        }

        INDArray mask = generateMask(inputs);
        currentBatch++;
        return new Batch(inputs, targets, mask);
    }

    private INDArray generateMask(INDArray inputs) {
        INDArray mask = Nd4j.ones(inputs.shape());

        for (int i = 0; i < inputs.size(0); i++) {
            for (int j = 0; j < inputs.size(1); j++) {
                if (inputs.getDouble(i, j) == tokenizer.getPadTokenId()) {
                    mask.putScalar(new int[]{i, j}, 0.0);
                }
            }
        }
        mask = mask.reshape(new int[]{(int) mask.size(0), 1, 1, (int) mask.size(1)});
        return mask;
    }

    public void init() {
        reset();
    }

    public void reset() {
        currentBatch = 0;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }
}
