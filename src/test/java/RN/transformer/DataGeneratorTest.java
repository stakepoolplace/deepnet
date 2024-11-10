package RN.transformer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.nd4j.linalg.api.ndarray.INDArray;

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
        generator = new DataGenerator(Arrays.asList("data sample"), Arrays.asList("target sample"), tokenizer, 1, maxTokenPerBatch);
    }

    @Test
    public void testTokenizerConversion() {
        // Initialisez le vocabulaire
        List<String> vocab = Arrays.asList("data", "sample", "target", "<PAD>", "<UNK>");
        Tokenizer tokenizer = new Tokenizer(vocab, 50);

        // Vérifiez le mappage tokenToId
        System.out.println("Mapping tokenToId: " + tokenizer.getTokenToId());

        // Définissez des tokens
        List<String> tokens = Arrays.asList("data", "sample");

        // Convertissez les tokens en IDs
        INDArray idsArray = tokenizer.tokensToINDArray(tokens);
        //System.out.println("IDs Array: " + Arrays.toString(idsArray.toIntVector()));

        // Convertissez les IDs en tokens
        List<Integer> ids = Arrays.stream(idsArray.toIntVector())
                                .boxed()
                                .collect(Collectors.toList());
        String tokensText = tokenizer.idsToTokens(ids);
        System.out.println("Tokens Text: " + tokensText);

        // Assertions
        Assert.assertArrayEquals("Data IDs should match expected", new int[]{4, 5}, idsArray.toIntVector());
        Assert.assertEquals("Tokens text should match", "data sample", tokensText);
    }


    @Test
    public void testDataAndTargetAssignment() throws IOException {
        // Initialisez le vocabulaire avec tous les mots nécessaires
        List<String> vocab = Arrays.asList("data", "sample", "target", "<PAD>", "<UNK>");
        Tokenizer tokenizer = new Tokenizer(vocab, 50);

        // Tokenize les phrases en listes de tokens individuels
        List<String> dataTokens = tokenizer.tokenize("data sample");      // ["data", "sample"]
        List<String> targetTokens = tokenizer.tokenize("target sample");  // ["target", "sample"]

        // Initialiser un Batch avec les tokens
        Batch batch = new Batch(dataTokens, targetTokens, tokenizer);

        // Vérifications de non-nullité
        Assert.assertNotNull("Batch should not be null", batch);
        Assert.assertNotNull("Data batch should not be null", batch.getData());
        Assert.assertNotNull("Target batch should not be null", batch.getTarget());

        // Affichez les IDs dans le Batch pour vérifier la conversion
        System.out.println("Data IDs in Batch: " + Arrays.toString(batch.getData().toIntVector()));
        System.out.println("Target IDs in Batch: " + Arrays.toString(batch.getTarget().toIntVector()));

        // Convertir les données `INDArray` en texte
        List<Integer> dataIds = Arrays.stream(batch.getData().toIntVector())
                                      .boxed()
                                      .collect(Collectors.toList());
        List<Integer> targetIds = Arrays.stream(batch.getTarget().toIntVector())
                                        .boxed()
                                        .collect(Collectors.toList());

        String dataText = tokenizer.idsToTokens(dataIds);
        String targetText = tokenizer.idsToTokens(targetIds);

        // Afficher les résultats pour vérification
        System.out.println("Data text: " + dataText);
        System.out.println("Target text: " + targetText);

        // Vérifications finales
        Assert.assertEquals("Data batch should contain expected tokens", "data sample", dataText);
        Assert.assertEquals("Target batch should contain expected tokens", "target sample", targetText);
    }
    
    

    @After
    public void tearDown() throws IOException {
        Files.deleteIfExists(dataFile);
        Files.deleteIfExists(targetFile);
    }
}
