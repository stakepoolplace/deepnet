package RN.transformer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class trainTest {
    public static void main(String[] args) throws IOException {
        // Configuration du Tokenizer
        Tokenizer tokenizer = new Tokenizer();
        tokenizer.addToken("start");
        tokenizer.addToken("end");
        tokenizer.addToken("hello");
        tokenizer.addToken("world");
        tokenizer.addToken("pad");
        tokenizer.buildVocab();
        
        // Données d'entraînement
        List<String> data = Arrays.asList("hello");
        List<String> targets = Arrays.asList("hello");
        
        // Configuration du DataGenerator
        int batchSize = 1;
        int sequenceLength = 3; // "start" + "hello" + "end"
        DataGenerator dataGenerator = new DataGenerator(data, targets, tokenizer, batchSize, sequenceLength);
        
        // Initialisation du Modèle Transformer
        int dModel = 16;
        Encoder encoder = new Encoder(dModel); // Assurez-vous que ces classes sont correctement implémentées
        Decoder decoder = new Decoder(dModel);
        TransformerModel model = new TransformerModel(encoder, decoder, tokenizer, dModel);
        model.initializeEmbeddings(tokenizer.getVocabSize(), dModel);
        model.addCombinedParameters();
        
        // Initialisation de l'Optimiseur avec un learning rate très bas
        CustomAdamOptimizer optimizer = new CustomAdamOptimizer(0.00001f, dModel, 50, model.getCombinedParameters());
        model.setOptimizer(optimizer);
        
        // Entraînement sur quelques epochs
        for (int epoch = 1; epoch <= 10; epoch++) {
            float avgLoss = model.trainEpoch(dataGenerator);
            System.out.printf("Epoch %d - Average Loss: %.6f%n", epoch, avgLoss);
        }
        
        // Inférence
        String prompt = "hello";
        String generated = model.infer(prompt, 3);
        System.out.println("Generated: " + generated);
    }
}

