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
    public void testSimpleTransformerTraining() throws IOException, ClassNotFoundException {
        // Configuration simplifiée
        int numLayers = 2;
        int dModel = 16;
        int numHeads = 2;
        int dff = 1024;
        double dropoutRate = 0.0;
        int vocabSize = 16;
        int maxSequenceLength = 5;
        float learningRate = 0.0001f;
        int warmupSteps = 10;
    
        // Initialiser le Tokenizer avec un vocabulaire simple
        List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world", "input", "output", "RN", "test", "chat", "chien", "oiseau", "rat", "camembert", "Sophie");
        Tokenizer tokenizer = new Tokenizer(vocab, dModel, maxSequenceLength);
        INDArray pretrainedEmbeddings = Nd4j.randn(vocabSize, dModel).divi(Math.sqrt(dModel));
        tokenizer.setPretrainedEmbeddings(pretrainedEmbeddings);
        
        // Initialiser le modèle Transformer avec le Tokenizer personnalisé
        TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, learningRate, warmupSteps);
        
        // Créer un jeu de données synthétique (entrée = cible)
        List<String> inputSentences = Arrays.asList("hello");
        List<String> targetSentences = Arrays.asList("hello");
        
        // Convertir les phrases en IDs de tokens
        List<Integer> inputIds = tokenizer.tokensToIds(tokenizer.tokenize(inputSentences.get(0)));
        List<Integer> targetIds = tokenizer.tokensToIds(tokenizer.tokenize(targetSentences.get(0)));
        
        // Ajouter les tokens spéciaux si nécessaire (par exemple, <START>, <END>)
        // inputIds.add(0, tokenizer.getStartTokenId());
        // targetIds.add(0, tokenizer.getStartTokenId());
        // targetIds.add(tokenizer.getEndTokenId());
        
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
        DataGenerator dataGenerator = new DataGenerator(
            Arrays.asList("hello"),
            Arrays.asList("hello"),
            tokenizer,
            1, // batchSize
            targetIds.size() // sequenceLength
        );
        
        // Initialiser l'optimiseur avec un taux d'apprentissage adapté
        transformer.optimizer = new CustomAdamOptimizer(learningRate, dModel, warmupSteps, transformer.getCombinedParameters());
        
        // Entraîner le modèle sur plusieurs epochs
        int epochs = 50;
        float previousLoss = Float.MAX_VALUE;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            float averageLoss = transformer.trainEpoch(dataGenerator);
            System.out.println("Epoch " + epoch + " - Average Loss: " + averageLoss);
            
            // Vérifier que la perte diminue
            //assertTrue("La perte devrait diminuer à chaque epoch.", averageLoss <= previousLoss);
            previousLoss = averageLoss;
            
            // Réinitialiser le DataGenerator pour le prochain epoch
            dataGenerator.reset();
        }
        
        // Marquer le modèle comme entraîné
        transformer.setTrained(true);
        
        // Effectuer une inférence sur la même entrée
        String prompt = "hello";
        String inferredOutput = transformer.infer(prompt, 3);
        System.out.println("Inferred Output: " + inferredOutput);
        
        // Afficher les relations entre les tokens après l'entraînement
        List<String> inputTokens = tokenizer.tokenize(prompt); // Ex: ["hello", "world", "input"]
        transformer.displayAttentionRelations(inputTokens);

        // Vérifier que l'inférence correspond à la cible attendue
        assertEquals("L'inférence devrait correspondre à la cible.", targetSentences.get(0), inferredOutput);

    }


    @Test
    public void testSimpleTransformerSimplified() throws IOException, ClassNotFoundException {
        // Configuration simplifiée
        int numLayers = 1;
        int dModel = 4;
        int numHeads = 2;
        int dff = 16;
        double dropoutRate = 0.0;
        int vocabSize = 5;
        int maxSequenceLength = 4;
        float learningRate = 0.001f;
        int warmupSteps = 10;
    
        // Initialiser le Tokenizer avec un vocabulaire très simple
        List<String> vocab = Arrays.asList("<PAD>", "<START>", "<END>", "hello", "world");
        Tokenizer tokenizer = new Tokenizer(vocab, dModel, maxSequenceLength);
        INDArray pretrainedEmbeddings = Nd4j.randn(vocabSize, dModel).divi(Math.sqrt(dModel));
        tokenizer.setPretrainedEmbeddings(pretrainedEmbeddings);
    
        // Initialiser le modèle Transformer avec le Tokenizer personnalisé
        TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, learningRate, warmupSteps);
    
        // Initialiser l'optimiseur
        transformer.initializeOptimizer(learningRate, warmupSteps);
    
        // Créer un jeu de données synthétique (entrée = cible)
        List<String> inputSentences = Arrays.asList("hello");
        List<String> targetSentences = Arrays.asList("world");
    
        // Convertir les phrases en IDs de tokens
        List<Integer> inputIds = tokenizer.tokensToIds(tokenizer.tokenize(inputSentences.get(0))); // [<START>, hello, <END>, <PAD>, ...]
        List<Integer> targetIds = tokenizer.tokensToIds(tokenizer.tokenize(targetSentences.get(0))); // [<START>, world, <END>, <PAD>, ...]
    
        // Convertir les listes en INDArray [batchSize, seqLength]
        INDArray input = Nd4j.create(DataType.INT, 1, inputIds.size());
        INDArray target = Nd4j.create(DataType.INT, 1, targetIds.size());
    
        for (int i = 0; i < inputIds.size(); i++) {
            input.putScalar(new int[] { 0, i }, inputIds.get(i));
        }
    
        for (int i = 0; i < targetIds.size(); i++) {
            target.putScalar(new int[] { 0, i }, targetIds.get(i));
        }
    
        // Créer un DataGenerator avec un seul batch
        DataGenerator dataGenerator = new DataGenerator(
                inputSentences,
                targetSentences,
                tokenizer,
                1, // batchSize
                targetIds.size() // sequenceLength
        );

        tokenizer.printVocabulary();
    
        // Entraîner le modèle sur plusieurs epochs
        int epochs = 50;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            float averageLoss = transformer.trainEpoch(dataGenerator);
            System.out.println("Epoch " + epoch + " - Average Loss: " + averageLoss);
            dataGenerator.reset();
        }
    
        // Marquer le modèle comme entraîné
        transformer.setTrained(true);
    
        // Effectuer une inférence sur la même entrée et stocker les poids d'attention
        String prompt = "hello";
        String inferredOutput = transformer.infer(prompt, maxSequenceLength); // Ajuster la longueur maximale si nécessaire
        System.out.println("Inferred Output: " + inferredOutput);
    

        // Afficher les relations entre les tokens après l'inférence sous forme de tableau croisé
        List<String> inputTokens = tokenizer.tokenize(prompt); // Ex: ["<START>", "hello", "<END>", "<PAD>"]
        transformer.displayAttentionRelations(inputTokens);

        // Vérifier que l'inférence correspond à la cible attendue
        assertEquals("L'inférence devrait correspondre à la cible.", targetSentences.get(0), inferredOutput);
    
    }
    




    @Test
    public void testTransformerTrainingAndInference() throws IOException, ClassNotFoundException {
        // Configuration du modèle
        int numLayers = 1;
        int dModel = 6;      // Pour simplifier, dModel = numHeads * depth (ex: 2 * 3)
        int numHeads = 2;
        int dff = 128;
        double dropoutRate = 0.0;
        int vocabSize = 6;  // Taille du vocabulaire pour l'exemple
        float learningRate = 0.001f;
        int epochs = 100;
        int maxSequenceLength = 5;
        int warmupSteps = 10;

        // Initialiser le Tokenizer avec un vocabulaire simple
        List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world");
        Tokenizer tokenizer = new Tokenizer(vocab, dModel, maxSequenceLength);
        INDArray pretrainedEmbeddings = Nd4j.randn(vocabSize, dModel).divi(Math.sqrt(dModel));
        tokenizer.setPretrainedEmbeddings(pretrainedEmbeddings);
        
        // Initialiser le modèle Transformer avec le Tokenizer personnalisé
        TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer, learningRate, warmupSteps);
        
        // Créer un jeu de données synthétique (entrée = cible)
        List<String> inputSentences = Arrays.asList("hello world", "hello world");
        List<String> targetSentences = Arrays.asList("world hello", "world hello");

        
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
        DataGenerator dataGenerator = new DataGenerator(
            Arrays.asList("hello world", "hello world"),
            Arrays.asList("world hello", "world hello"),
            tokenizer,
            2, // batchSize
            targetIds.size() // sequenceLength
        );
        
        // Initialiser l'optimiseur avec un taux d'apprentissage adapté
        transformer.optimizer = new CustomAdamOptimizer(learningRate, dModel, 1000, transformer.getCombinedParameters());
        
        // Entraîner le modèle sur plusieurs epochs
        
        float previousLoss = Float.MAX_VALUE;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            float averageLoss = transformer.trainEpoch(dataGenerator);
            System.out.println("Epoch " + epoch + " - Average Loss: " + averageLoss);
            
            // Vérifier que la perte diminue
            //assertTrue("La perte devrait diminuer à chaque epoch.", averageLoss <= previousLoss);
            previousLoss = averageLoss;
            
            // Réinitialiser le DataGenerator pour le prochain epoch
            dataGenerator.reset();
        }
        
        // Marquer le modèle comme entraîné
        transformer.setTrained(true);
        
        // Effectuer une inférence sur la même entrée
        String prompt = "hello world";
        String inferredOutput = transformer.infer(prompt, maxSequenceLength);
        System.out.println("Inferred Output: " + inferredOutput);
        
        // Vérifier que l'inférence correspond à la cible attendue
        assertEquals("L'inférence devrait correspondre à la cible.", targetSentences.get(0), inferredOutput);
        
        // Effectuer un Gradient Check
        //performGradientCheck(transformer, input, target);
    }

    /**
     * Effectue un gradient check sur le modèle Transformer.
     * 
     * @param transformer Instance du TransformerModel
     * @param input       Entrée sous forme d'indices de tokens [batchSize, seqLength]
     * @param target      Cible sous forme d'indices de tokens [batchSize, seqLength]
     */
    // public void performGradientCheck(TransformerModel transformer, INDArray input, INDArray target) {
    //     double epsilon = 1e-5;
    //     double tolerance = 1e-4;
    //     int maxDiffCount = 0;
    //     int totalChecks = 0;
        
    //     // Calculer les gradients analytiques
    //     System.out.println("Calcul des gradients analytiques...");
    //     double loss = transformer.calculateCrossEntropyLossAndGradient(input, target);
    //     List<INDArray> analyticalGradients = transformer.getGradients();
    //     List<INDArray> parameters = transformer.getParameters();
        
    //     // Itérer sur chaque paramètre
    //     for (int p = 0; p < parameters.size(); p++) {
    //         INDArray param = parameters.get(p);
    //         INDArray gradAnalytic = analyticalGradients.get(p);
            
    //         System.out.println("Vérification du gradient pour le paramètre " + p + " avec forme " + Arrays.toString(param.shape()));
            
    //         // Itérer sur chaque élément du paramètre
    //         for (int i = 0; i < param.length(); i++) {
    //             totalChecks++;
                
    //             // Sauvegarder la valeur originale
    //             double originalValue = param.getDouble(i);
                
    //             // Perturber positivement
    //             param.putScalar(i, originalValue + epsilon);
    //             double lossPlus = transformer.calculateCrossEntropyLoss(input, target);
                
    //             // Perturber négativement
    //             param.putScalar(i, originalValue - epsilon);
    //             double lossMinus = transformer.calculateCrossEntropyLoss(input, target);
                
    //             // Restaurer la valeur originale
    //             param.putScalar(i, originalValue);
                
    //             // Calculer le gradient numérique
    //             double numericGrad = (lossPlus - lossMinus) / (2 * epsilon);
                
    //             // Obtenir le gradient analytique
    //             double analyticGrad = gradAnalytic.getDouble(i);
                
    //             // Calculer la différence
    //             double diff = Math.abs(numericGrad - analyticGrad);
                
    //             // Vérifier si la différence dépasse la tolérance
    //             if (diff > tolerance) {
    //                 maxDiffCount++;
    //                 System.out.println("---- Gradient Check Failed ----");
    //                 System.out.println("Paramètre: " + p);
    //                 System.out.println("Indice: " + i);
    //                 System.out.println("Valeur originale: " + originalValue);
    //                 System.out.println("Gradient Numérique: " + numericGrad);
    //                 System.out.println("Gradient Analytique: " + analyticGrad);
    //                 System.out.println("Différence: " + diff);
    //                 // Vous pouvez choisir de continuer ou d'arrêter ici
    //                 // Pour cette implémentation, nous continuons pour afficher tous les échecs
    //             }
    //         }
    //     }
        
    //     // Résumé du Gradient Check
    //     if (maxDiffCount == 0) {
    //         System.out.println("Gradient Check Passed! Toutes les vérifications sont conformes.");
    //     } else {
    //         System.out.println("Gradient Check Completed with " + maxDiffCount + " différences dépassant la tolérance sur " + totalChecks + " vérifications.");
    //     }
    // }

    
}
