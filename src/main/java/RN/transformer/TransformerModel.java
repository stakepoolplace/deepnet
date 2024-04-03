package RN.transformer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TransformerModel {
    private boolean isTrained = false;
    private Encoder encoder;
    private Decoder decoder;
    private CustomAdamOptimizer optimizer;
    private Tokenizer tokenizer;
    private double dropoutRate = 0.1; // Exemple de taux de dropout fixe


    public TransformerModel() throws IOException {
    	
        int numLayers = 6;
        int dModel = 512;
        int numHeads = 8;
        int dff = 2048;
        int vocabSize = 10000; // Exemple hypothétique
        int maxSeqLength = 512; // Exemple hypothétique
        
        this.tokenizer = new Tokenizer(); // Supposé exister pour gérer la tokenisation
        
        // Créer une matrice d'embeddings pré-entraînée
        INDArray pretrainedEmbeddings = createPretrainedEmbeddings(vocabSize, dModel);
        
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, vocabSize, pretrainedEmbeddings, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize);
        this.optimizer = new CustomAdamOptimizer(0.001, 1000); // Initialisation hypothétique
    }
    
    private INDArray createPretrainedEmbeddings(int vocabSize, int dModel) throws IOException {
        // Vous devrez adapter ce code en fonction du format de vos embeddings pré-entraînés
        // Supposons que vos embeddings pré-entraînés sont stockés dans un fichier texte où chaque ligne représente un token ID suivi de ses embeddings

        String filePath = "chemin/vers/votre/fichier/embeddings.txt";
        List<String> lines = Files.readAllLines(Paths.get(filePath));
        
        // Créer une matrice pour stocker les embeddings
        INDArray embeddings = Nd4j.create(vocabSize, dModel);

        for (String line : lines) {
            String[] parts = line.split(" "); // Supposons que les embeddings sont séparés par des espaces
            int tokenId = Integer.parseInt(parts[0]);
            // Les embeddings sont les éléments restants de la ligne
            float[] embeddingValues = new float[dModel];
            for (int i = 1; i < parts.length; i++) {
                embeddingValues[i - 1] = Float.parseFloat(parts[i]);
            }
            // Assigner les embeddings à la ligne correspondante dans la matrice
            embeddings.putRow(tokenId, Nd4j.create(embeddingValues));
        }

        return embeddings;
    }    

    public void train(DataGenerator dataGenerator) throws IOException {

        for (int epoch = 0; epoch < 10; epoch++) {
        	int batchNum = 0;
            // Itération sur les batches générés par le générateur de données
            while (dataGenerator.hasNextBatch()) {
                
            	Batch batch = dataGenerator.nextBatch();
                
                // Tokenization du texte cible
            	String batchTarget = String.join("", batch.getTarget());
                List<String> targetTokens = tokenizer.tokenize(batchTarget);
                // Conversion des tokens cibles en IDs
                List<Integer> targetTokenIds = tokenizer.tokensToIds(targetTokens);
                
                // Tokenization du text source
            	String batchData = String.join("", batch.getData());
                List<String> dataTokens = tokenizer.tokenize(batchData);
                // Conversion des tokens sources en IDs
                List<Integer> dataTokenIds = tokenizer.tokensToIds(dataTokens);

	            // Supposons que 'encode' et 'decode' retournent une liste de logits par token
            	INDArray encoded = encoder.encode(dataTokenIds);
	            List<List<Float>> decodedLogits = decoder.decode(encoded);
	           	            
	            
	            
                // Calculez la perte à l'aide d'une méthode hypothétique calculateLoss
	            float loss = calculateLoss(decodedLogits, targetTokenIds);

                // Supposons que vous ayez des méthodes pour calculer les gradients par rapport à la perte
                INDArray encoderGradients = encoder.calculateGradients(loss);
                INDArray decoderGradients = decoder.calculateGradients(loss);

                // Mise à jour des paramètres de l'encodeur et du décodeur via l'optimiseur
                optimizer.update(encoder.getParameters(), encoderGradients);
                optimizer.update(decoder.getParameters(), decoderGradients);	            
	            
	            
	            
	            
	            
	            batchNum++;
            }
            
            dataGenerator.init();
        }

        isTrained = true;
    }



    public String infer(String prompt) {
        if (!isTrained) {
            throw new IllegalStateException("Le modèle doit être entraîné avant l'inférence.");
        }

        // Tokenisation et conversion du prompt en IDs
        List<String> promptTokens = tokenizer.tokenize(prompt);
        List<Integer> promptTokenIds = tokenizer.tokensToIds(promptTokens);

        // Encodage
        INDArray encodedPrompt = encoder.encode(promptTokenIds); // Supposons que la méthode encode retourne un INDArray

        // Décodeur : Préparation des arguments nécessaires
        // Supposons que encoderOutput est le même que encodedPrompt pour l'inférence simple
        // Les masques lookAheadMask et paddingMask sont initialisés à null pour l'exemple
        INDArray lookAheadMask = null;
        INDArray paddingMask = null;
        INDArray logits = decoder.forward(encodedPrompt, encodedPrompt, lookAheadMask, paddingMask);

        // Conversion des logits en IDs de tokens
        INDArray predictedTokenIds = Nd4j.argMax(logits, 2);

        // Conversion de INDArray en List<Integer>
        long[] shape = predictedTokenIds.shape();
        List<Integer> tokenIdsList = new ArrayList<>();
        for(int i = 0; i < shape[0]; i++) { 
            tokenIdsList.add(predictedTokenIds.getInt(i));
        }

        // Conversion des IDs de tokens prédits en texte
        String response = tokenizer.idsToTokens(tokenIdsList);

        return response;
    }






    public boolean isTrained() {
        return isTrained;
    }

    private float calculateLoss(List<List<Float>> decodedLogits, List<Integer> targetTokenIds) {
    	
        float loss = 0.0f;
        int N = targetTokenIds.size();

        for (int i = 0; i < N; i++) {
            int targetId = targetTokenIds.get(i);
            List<Float> logitsForToken = decodedLogits.get(i);
            
            float sumExpLogits = 0.0f;
            for (Float logit : logitsForToken) {
                sumExpLogits += Math.exp(logit);
            }
            
            float logSoftmaxForTarget = (float)Math.log(Math.exp(logitsForToken.get(targetId)) / sumExpLogits);
            loss += -logSoftmaxForTarget;
        }
        
        return loss / N;
    }
}
