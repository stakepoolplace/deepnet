package RN.transformer;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TransformerGradientCheck {
    public static void main(String[] args) {
        // Configuration simplifiée
        int numLayers = 1;
        int dModel = 6;
        int numHeads = 2;
        int dff = 12;
        double dropoutRate = 0.1;
        int vocabSize = 10;
        int maxSequenceLength = 50;
        float learningRate = 0.001f;

        // Initialiser le Tokenizer avec un vocabulaire simple
        List<String> vocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>", "hello", "world", "test", "input", "output", "RN");
        Tokenizer tokenizer = new Tokenizer(vocab, dModel, maxSequenceLength);
        INDArray pretrainedEmbeddings = Nd4j.randn(vocabSize, dModel).divi(Math.sqrt(dModel));
        tokenizer.setPretrainedEmbeddings(pretrainedEmbeddings);
        
        // Initialiser le modèle Transformer avec le Tokenizer personnalisé
        TransformerModel transformer = new TransformerModel(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize, tokenizer);
        
        // Créer un jeu de données synthétique (entrée = cible)
        List<String> inputSentences = Arrays.asList("hello");
        List<String> targetSentences = Arrays.asList("hello");
        
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
            Arrays.asList("hello"),
            Arrays.asList("hello"),
            tokenizer,
            1, // batchSize
            targetIds.size() // sequenceLength
        );
        
        // Initialiser l'optimiseur avec un taux d'apprentissage adapté
        transformer.optimizer = new CustomAdamOptimizer(learningRate, dModel, 1000, transformer.getCombinedParameters());
        
        // Effectuer un forward pass
        float loss = transformer.calculateCrossEntropyLossAndGradient(input, target);
        System.out.println("Initial Loss: " + loss);
        
        // Gradient Checking
        performGradientCheck(transformer, input, target);
    }
    
    public static void performGradientCheck(TransformerModel transformer, INDArray input, INDArray target) {
        // Calcul des gradients analytiques
        transformer.forward(input);
        transformer.calculateCrossEntropyLossAndGradient(input, target);
        List<INDArray> grads = transformer.getGradients();
        List<INDArray> params = transformer.getParameters();
        
        double epsilon = 1e-5;
        double tolerance = 1e-4;
        
        for (int i = 0; i < params.size(); i++) {
            INDArray param = params.get(i);
            INDArray gradAnalytic = grads.get(i);
            INDArray gradNumerical = Nd4j.zeros(param.shape());
            
            // Calcul des gradients numériques
            for (int j = 0; j < param.length(); j++) {
                double originalValue = param.getDouble(j);
                
                // Perturber positivement
                param.putScalar(j, originalValue + epsilon);
                double lossPlus = transformer.calculateCrossEntropyLossAndGradient(input, target);
                
                // Perturber négativement
                param.putScalar(j, originalValue - epsilon);
                double lossMinus = transformer.calculateCrossEntropyLossAndGradient(input, target);
                
                // Restaurer la valeur originale
                param.putScalar(j, originalValue);
                
                // Calculer le gradient numérique
                double numericGrad = (lossPlus - lossMinus) / (2 * epsilon);
                gradNumerical.putScalar(j, numericGrad);
            }
            
            // Comparer les gradients
            INDArray diff = Transforms.abs(gradAnalytic.sub(gradNumerical));
            double maxDifference = diff.maxNumber().doubleValue();
            System.out.println("Gradient Check for Param " + i + ": Max Difference = " + maxDifference);
            if (maxDifference > tolerance) {
                System.out.println("Gradient Check Failed for Param " + i);
            } else {
                System.out.println("Gradient Check Passed for Param " + i);
            }
        }
    }
    
    // Fonction de perte simple (somme des sorties)
    private static double lossFunction(INDArray output) {
        return output.sumNumber().doubleValue();
    }
}
