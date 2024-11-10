package RN.transformer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class MockDataGenerator extends DataGenerator {

    private int batchCount = 0;
    private int maxBatches;
    private Tokenizer tokenizer;

    public MockDataGenerator(List<String> data, List<String> targets, Tokenizer tokenizer, int batchSize, int maxTokensPerBatch, int maxBatches) {
        super(data, targets, tokenizer, batchSize, maxTokensPerBatch);
        this.maxBatches = maxBatches;
        this.tokenizer = tokenizer;
    }

    @Override
    public boolean hasNextBatch() {
        return batchCount < maxBatches;
    }

    @Override
    public Batch nextBatch() {
        if (!hasNextBatch()) {
            return null;
        }
        batchCount++;
        
        // Exemple de données pour créer un batch
        List<String> data = Arrays.asList("hello world");
        List<String> target = Arrays.asList("hello output");

        // Créer le batch avec le tokenizer pour convertir en INDArray
        return new Batch(data, target, tokenizer);
    }

    @Override
    public void init() {
        batchCount = 0;
    }
}
