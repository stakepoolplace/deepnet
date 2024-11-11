package RN.transformer;

import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.commons.lang3.tuple.Pair;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Test unitaire pour vérifier l'entraînement et l'inférence du modèle Transformer.
 */
public class TransformerInferenceTest {

    /**
     * Test d'entraînement et d'inférence sur un jeu de données synthétique.
     */
    @Test
    public void testTransformerTrainingAndInference() throws IOException, ClassNotFoundException {
        // Configuration du modèle
        int numLayers = 2;
        int dModel = 6;      // Pour simplifier, dModel = numHeads * depth (ex: 2 * 3)
        int numHeads = 2;
        int dff = 12;
        double dropoutRate = 0.1;
        int vocabSize = 10;  // Taille du vocabulaire pour l'exemple
        
        float learningRate = 0.0001f;


        // Initialiser le Tokenizer avec un vocabulaire simple
        List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world", "test", "input", "output", "RN");
        Tokenizer tokenizer = new Tokenizer(vocab, dModel);
        INDArray pretrainedEmbeddings = Nd4j.randn(vocabSize, dModel).divi(Math.sqrt(dModel));
        tokenizer.setPretrainedEmbeddings(pretrainedEmbeddings);
        
        // Initialiser le modèle Transformer avec le Tokenizer personnalisé
        TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer);
        
        // Créer un jeu de données synthétique (entrée = cible)
        // Par exemple, "hello world" devrait être prédit comme "hello world"
        List<String> inputSentences = Arrays.asList("hello world", "input output");
        List<String> targetSentences = Arrays.asList("hello world", "input output");

        
        // Convertir les phrases en IDs de tokens
        List<Integer> inputIds = tokenizer.tokensToIds(tokenizer.tokenize(inputSentences.get(0)));
        List<Integer> targetIds = tokenizer.tokensToIds(tokenizer.tokenize(targetSentences.get(0)));
        
        // Ajouter les tokens spéciaux si nécessaire (par exemple, <START>, <END>)
        inputIds.add(0, tokenizer.getStartTokenId());
        targetIds.add(0, tokenizer.getStartTokenId());
        targetIds.add(tokenizer.getEndTokenId());
        
        // Convertir les listes en INDArray [batchSize, seqLength]
        INDArray input = Nd4j.create(DataType.INT, 1, inputIds.size());
        INDArray target = Nd4j.create(DataType.INT, 1, targetIds.size());
        
        for (int i = 0; i < inputIds.size(); i++) {
            input.putScalar(new int[] {0, i}, inputIds.get(i));
        }
        
        for (int i = 0; i < targetIds.size(); i++) {
            target.putScalar(new int[] {0, i}, targetIds.get(i));
        }
        
        // Créer un DataGenerator avec un seul batch
        Batch batch = new Batch(input, target, transformer.createPaddingMask(input));
        DataGenerator dataGenerator = new DataGenerator(
            Arrays.asList("hello world", "input output"),
            Arrays.asList("hello world", "input output"),
            tokenizer,
            2, // batchSize
            targetIds.size() // sequenceLength
        );
        
        // Initialiser l'optimiseur avec un taux d'apprentissage adapté
        transformer.optimizer = new CustomAdamOptimizer(learningRate, dModel, 1000, transformer.getCombinedParameters());
        
        // Entraîner le modèle sur plusieurs epochs
        int epochs = 200;
        float previousLoss = Float.MAX_VALUE;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            float averageLoss = transformer.trainEpoch(dataGenerator);
            System.out.println("Epoch " + epoch + " - Average Loss: " + averageLoss);
            
            // Vérifier que la perte diminue
            assertTrue("La perte devrait diminuer à chaque epoch.", averageLoss <= previousLoss);
            previousLoss = averageLoss;
            
            // Réinitialiser le DataGenerator pour le prochain epoch
            dataGenerator.reset();
        }
        
        // Marquer le modèle comme entraîné
        transformer.setTrained(true);
        
        // Effectuer une inférence sur la même entrée
        String prompt = "hello world";
        String inferredOutput = transformer.infer(prompt, targetIds.size());
        System.out.println("Inferred Output: " + inferredOutput);
        
        // Vérifier que l'inférence correspond à la cible attendue
        assertEquals("L'inférence devrait correspondre à la cible.", targetSentences.get(0), inferredOutput);
    }
}
