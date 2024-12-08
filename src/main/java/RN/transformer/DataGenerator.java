// DataGenerator.java
package RN.transformer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DataGenerator {
    private List<List<Integer>> inputSequences;
    private List<List<Integer>> targetSequences;
    private final Tokenizer tokenizer;
    private final int batchSize;
    private final int sequenceLength;
    private int currentIndex;

    public DataGenerator(List<String> inputs, List<String> targets, Tokenizer tokenizer, int batchSize, int sequenceLength) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Les listes d'entrée et de cible doivent avoir la même taille");
        }
        
        this.tokenizer = tokenizer;
        this.batchSize = Math.min(batchSize, inputs.size());
        this.sequenceLength = sequenceLength;
        this.currentIndex = 0;

        // Prétraiter les entrées
        this.inputSequences = inputs.stream()
            .map(text -> tokenizer.tokensToIds(tokenizer.tokenize(text)))
            .collect(Collectors.toList());
            
        // Prétraiter les cibles
        this.targetSequences = targets.stream()
            .map(text -> tokenizer.tokensToIds(tokenizer.tokenize(text)))
            .collect(Collectors.toList());
    }

    public Batch getNextBatch() {
        int endIdx = Math.min(currentIndex + batchSize, inputSequences.size());
        if (currentIndex >= inputSequences.size()) {
            return null;
        }
        
        List<List<Integer>> batchInputs = inputSequences.subList(currentIndex, endIdx);
        List<List<Integer>> batchTargets = targetSequences.subList(currentIndex, endIdx);
        currentIndex += batchSize;
        
        // Convertir en INDArray
        INDArray inputArray = convertToINDArray(batchInputs);
        INDArray targetArray = convertToINDArray(batchTargets);
        
        return new Batch(inputArray, targetArray, tokenizer);
    }

    public boolean hasNextBatch() {
        return currentIndex < inputSequences.size();
    }

    public void reset() {
        currentIndex = 0;
    }

    private INDArray convertToINDArray(List<List<Integer>> sequences) {
        INDArray array = Nd4j.zeros(DataType.INT, sequences.size(), sequenceLength);
        for (int i = 0; i < sequences.size(); i++) {
            List<Integer> seq = sequences.get(i);
            for (int j = 0; j < Math.min(seq.size(), sequenceLength); j++) {
                array.putScalar(new int[]{i, j}, seq.get(j).intValue());
            }
        }
        return array;
    }
}
