package RN.transformer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataGenerator {
    private BufferedReader dataReader;
    private BufferedReader targetReader;
    private int batchSize;
	private String dataFilePath;
	private String targetFilePath;

    public DataGenerator(String dataFilePath, String targetFilePath, int batchSize) throws IOException {
    	this.dataFilePath = dataFilePath;
    	this.targetFilePath = targetFilePath;
        this.dataReader = new BufferedReader(new FileReader(dataFilePath));
        this.targetReader = new BufferedReader(new FileReader(targetFilePath));
        this.batchSize = batchSize;
    }

    public boolean hasNextBatch() throws IOException {
        return dataReader.ready() && targetReader.ready();
    }

    public Batch nextBatch() throws IOException {
        List<String> dataBatch = new ArrayList<>();
        List<String> targetBatch = new ArrayList<>();

        for (int i = 0; i < batchSize && hasNextBatch(); i++) {
            dataBatch.add(dataReader.readLine());
            targetBatch.add(targetReader.readLine());
        }

        return new Batch(dataBatch, targetBatch);
    }

    public void close() throws IOException {
        dataReader.close();
        targetReader.close();
    }
    
    public void init() throws IOException {
        dataReader = new BufferedReader(new FileReader(dataFilePath));
        targetReader = new BufferedReader(new FileReader(targetFilePath));
    }    
}
