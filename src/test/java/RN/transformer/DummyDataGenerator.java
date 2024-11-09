package RN.transformer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

class DummyDataGenerator extends DataGenerator {
    
    private int maxBatchesPerEpoch;
    private int currentBatch;

    /**
     * Constructeur de DummyDataGenerator avec un nombre limité de batches par epoch.
     *
     * @param dataPath         Chemin vers les données (non utilisé dans ce dummy).
     * @param targetPath       Chemin vers les cibles (non utilisé dans ce dummy).
     * @param tokenizer        Instance du tokenizer.
     * @param batchSize        Taille du batch.
     * @param maxTokensPerBatch Nombre maximum de tokens par batch.
     * @param maxBatchesPerEpoch Nombre maximum de batches par epoch.
     * @throws IOException Si une erreur d'entrée/sortie survient.
     */
    public DummyDataGenerator(String dataPath, String targetPath, Tokenizer tokenizer, int batchSize, int maxTokensPerBatch, int maxBatchesPerEpoch) throws IOException {
        super(dataPath, targetPath, tokenizer, batchSize, maxTokensPerBatch);
        this.maxBatchesPerEpoch = maxBatchesPerEpoch;
        this.currentBatch = 0;
    }

    @Override
    public boolean hasNextBatch() {
        return currentBatch < maxBatchesPerEpoch;
    }

    @Override
    public Batch nextBatch() {
        currentBatch++;
        List<String> dummyData = Arrays.asList("This is a dummy sentence.");
        List<String> dummyTarget = Arrays.asList("Ceci est une phrase fictive.");
        return new Batch(dummyData, dummyTarget);
    }

    @Override
    public void init() {
        currentBatch = 0;
    }
}
