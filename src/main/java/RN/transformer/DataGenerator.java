package RN.transformer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataGenerator {
    private BufferedReader dataReader;
    private BufferedReader targetReader;
    protected Tokenizer tokenizer;
    private int batchSize;
    private int maxTokensPerBatch;
	private String targetFilePath;
	private String dataFilePath;

    public DataGenerator(String dataFilePath, String targetFilePath, Tokenizer tokenizer, int batchSize, int maxTokensPerBatch) throws IOException {
        this.dataReader = new BufferedReader(new FileReader(dataFilePath));
        this.targetReader = new BufferedReader(new FileReader(targetFilePath));
        this.targetFilePath = targetFilePath;
        this.dataFilePath = dataFilePath;
        this.tokenizer = tokenizer;
        this.batchSize = batchSize;
        this.maxTokensPerBatch = maxTokensPerBatch;
    }

    public boolean hasNextBatch() throws IOException {
        return dataReader.ready() && targetReader.ready();
    }

    // public Batch nextBatch() throws IOException {
    //     List<String> dataBatch = new ArrayList<>();
    //     List<String> targetBatch = new ArrayList<>();
    //     StringBuilder dataBuffer = new StringBuilder();
    //     StringBuilder targetBuffer = new StringBuilder();

    //     while (dataBatch.size() < batchSize && hasNextBatch()) {
    //         int dataChar;
    //         while ((dataChar = dataReader.read()) != -1) {
    //             dataBuffer.append((char) dataChar);
    //             if (dataBuffer.length() >= maxTokensPerBatch) break;
    //         }

    //         int targetChar;
    //         while ((targetChar = targetReader.read()) != -1) {
    //             targetBuffer.append((char) targetChar);
    //             if (targetBuffer.length() >= maxTokensPerBatch) break;
    //         }

    //         if (dataBuffer.length() > 0 && targetBuffer.length() > 0) {
    //             List<String> dataTokens = tokenizer.tokenize(dataBuffer.toString());
    //             List<String> targetTokens = tokenizer.tokenize(targetBuffer.toString());
    //             dataBatch.add(String.join(" ", dataTokens));
    //             targetBatch.add(String.join(" ", targetTokens));
    //             dataBuffer = new StringBuilder(); // Réinitialiser les buffers pour le prochain segment
    //             targetBuffer = new StringBuilder();
    //         }
    //     }

    //     return new Batch(dataBatch, targetBatch);
    // }

    
    public Batch nextBatch() throws IOException {
        List<String> dataBatch = new ArrayList<>();
        List<String> targetBatch = new ArrayList<>();
    
        for (int i = 0; i < batchSize; i++) {
            String dataLine = dataReader.readLine();
            String targetLine = targetReader.readLine();
    
            if (dataLine == null || targetLine == null) {
                break;
            }
    
            List<String> dataTokens = tokenizer.tokenize(dataLine);
            List<String> targetTokens = tokenizer.tokenize(targetLine);
            dataBatch.add(String.join(" ", dataTokens));
            targetBatch.add(String.join(" ", targetTokens));
        }
    
        return new Batch(dataBatch, targetBatch);
    }
    
    

    public void close() throws IOException {
        dataReader.close();
        targetReader.close();
    }

    public void init() throws IOException {
        this.dataReader = new BufferedReader(new FileReader(dataFilePath));
        this.targetReader = new BufferedReader(new FileReader(targetFilePath));
    }    
}
