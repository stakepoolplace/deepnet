package RN.transformer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TransformerModel {
    private boolean isTrained = false;
    private Encoder encoder;
    private Decoder decoder;
    private CustomAdamOptimizer optimizer;
    private Tokenizer tokenizer;

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
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, vocabSize);
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

    public void train(DataGenerator dataGenerator) {

        for (int epoch = 0; epoch < 10; epoch++) {
        	int batchNum = 0;
            // Itération sur les batches générés par le générateur de données
            while (dataGenerator.hasNextBatch()) {
                
            	Batch batch = dataGenerator.nextBatch();
                
                // Tokenization du texte cible
                List<String> targetTokens = tokenizer.tokenize(batch.getTarget());
                // Conversion des tokens cibles en IDs
                List<Integer> targetTokenIds = tokenizer.tokensToIds(targetTokens);

	            // Supposons que 'encode' et 'decode' retournent une liste de logits par token
	            List<List<Float>> encoded = encoder.encode(batch.getData());
	            List<List<Float>> decodedLogits = decoder.decode(encoded);
	            
	            float loss = calculateLoss(decodedLogits, targetTokenIds);
	            optimizer.updateModel(encoder, decoder, loss, batchNum, epoch);
	            
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

        List<List<Float>> encodedPrompt = encoder.encode(prompt);
        List<Float> decodedOutput = decoder.decode(encodedPrompt).get(0); // Simplification

        String response = tokenizer.idsToTokens(decodedOutput);
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
