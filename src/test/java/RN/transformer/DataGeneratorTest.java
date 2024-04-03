package RN.transformer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

public class DataGeneratorTest {

    private Path dataFile;
    private Path targetFile;
    private DataGenerator generator;
    private Tokenizer tokenizer;

    @Before
    public void setUp() throws IOException {
        // Create temporary files for data and target
        dataFile = Files.createTempFile("testData", ".txt");
        targetFile = Files.createTempFile("testTarget", ".txt");
        
        // Write some dummy data to files
        Files.write(dataFile, "data sample".getBytes());
        Files.write(targetFile, "target sample".getBytes());
        
        // Mocking the tokenizer
        tokenizer = Mockito.mock(Tokenizer.class);
        int maxTokenPerBatch = 13;
        Mockito.when(tokenizer.tokenize("data sample")).thenReturn(List.of("data", "sample"));
        Mockito.when(tokenizer.tokenize("target sample")).thenReturn(List.of("target", "sample"));

        // Initialize DataGenerator
        generator = new DataGenerator(dataFile.toString(), targetFile.toString(), tokenizer, 1, maxTokenPerBatch);
    }

    @Test
    public void testNextBatch() throws IOException {
        Batch batch = generator.nextBatch();

        Assert.assertNotNull("Batch should not be null", batch);
        Assert.assertFalse("Data batch should not be empty", batch.getData().isEmpty());
        Assert.assertFalse("Target batch should not be empty", batch.getTarget().isEmpty());
        Assert.assertEquals("Data batch should contain expected tokens", "data sample", batch.getData().get(0));
        Assert.assertEquals("Target batch should contain expected tokens", "target sample", batch.getTarget().get(0));
    }

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(dataFile);
        Files.deleteIfExists(targetFile);
    }
}
