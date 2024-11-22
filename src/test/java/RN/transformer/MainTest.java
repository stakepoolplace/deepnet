package RN.transformer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MainTest {
    public static void main(String[] args) {
        try {
            // Définir le vocabulaire et initialiser le Tokenizer
            List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
            int dModel = 4;
            int numLayers = 1;
            int numHeads = 1;
            int dff = 8;
            double dropoutRate = 0.0; // Désactiver le dropout pour la simulation
            
            Tokenizer tokenizer = new Tokenizer(vocab, dModel,3);
            TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocab.size(), tokenizer, 0.001f, 10);
            
            // Créer le batch de données
            List<String> inputTokens = Arrays.asList("<START>", "hello", "world");
            List<String> targetTokens = Arrays.asList("hello", "world", "<END>");
            
            List<Integer> inputIds = transformer.tokenizer.tokensToIds(inputTokens); // [2, 4, 5]
            List<Integer> targetIds = transformer.tokenizer.tokensToIds(targetTokens); // [4, 5, 3]
            
            INDArray inputData = Nd4j.create(new int[][] { inputIds.stream().mapToInt(i -> i).toArray() }); // [1, 3]
            INDArray targetData = Nd4j.create(new int[][] { targetIds.stream().mapToInt(i -> i).toArray() }); // [1, 3]
            
            Batch batch = new Batch(inputData, targetData, null);
            List<Batch> batchList = Arrays.asList(batch);
            DataGenerator dataGenerator = new DataGenerator(batchList);
            
            // Entraîner sur une seule epoch
            transformer.train(dataGenerator, 1);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}