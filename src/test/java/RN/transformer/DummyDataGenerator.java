package RN.transformer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

class DummyDataGenerator extends DataGenerator {
	
    public DummyDataGenerator(String dataPath, String targetPath, Tokenizer tokenizer, int batchSize, int maxTokensPerBatch) throws IOException {
        super(dataPath, targetPath, tokenizer, batchSize, maxTokensPerBatch);
    }

    @Override
    public boolean hasNextBatch() {
        // Simuler qu'il y a toujours au moins un batch disponible
        return true;
    }

    @Override
    public Batch nextBatch() {
        // Retourner un batch fictif
        List<String> dummyData = Arrays.asList("This is a dummy sentence.");
        List<String> dummyTarget = Arrays.asList("Ceci est une phrase fictive.");
        return new Batch(dummyData, dummyTarget);
    }
}
