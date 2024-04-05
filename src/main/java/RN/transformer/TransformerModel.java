package RN.transformer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TransformerModel {
    private boolean isTrained = false;
    public Encoder encoder;
    public Decoder decoder;
    public CustomAdamOptimizer optimizer;
    public Tokenizer tokenizer;
    private double dropoutRate = 0.1; // Exemple de taux de dropout fixe
    private static WordVectors wordVectors; // Chargé une fois, accessible statiquement
    private static int vocabSize = 0;
    private static INDArray meanVector = null;
    private static int dModel = 512;
    private static int numLayers = 6;
    private static int numHeads = 8;
    private static int dff = 2048;

    static {
        try {
            wordVectors = WordVectorSerializer.loadStaticModel(new File("pretrained-embeddings/model.bin"));
            vocabSize = wordVectors.vocab().numWords(); // Taille du vocabulaire Word2Vec
         // Calculer le vecteur moyen (à faire une seule fois, idéalement dans le constructeur ou une méthode d'initialisation)
            INDArray allVectors = Nd4j.create(vocabSize, dModel);
            for (int i = 0; i < vocabSize; i++) {
                String word = wordVectors.vocab().wordAtIndex(i);
                INDArray vector = wordVectors.getWordVectorMatrix(word);
                allVectors.putRow(i, vector);
            }
            meanVector = allVectors.mean(0); // Moyenne sur toutes les lignes pour obtenir un vecteur moyen

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    

    public TransformerModel() throws IOException {
        
        this.tokenizer = new Tokenizer(wordVectors); // Supposé exister pour gérer la tokenisation

        // Créer une matrice d'embeddings pré-entraînée
        INDArray pretrainedEmbeddings = createPretrainedEmbeddings(dModel);
        
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, vocabSize, pretrainedEmbeddings, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, vocabSize);
        
        // Calcul du nombre total de paramètres
        long totalParams = encoder.getNumberOfParameters() + decoder.getNumberOfParameters();
        
        this.optimizer = new CustomAdamOptimizer(0.001, 1000, totalParams); // Initialisation hypothétique
    }
    
 
    
    public INDArray createPretrainedEmbeddings(int dModel) {

        // Créer une matrice pour stocker les embeddings
        INDArray embeddings = Nd4j.create(vocabSize, dModel);
        
        // Pour chaque mot dans le vocabulaire du tokenizer
        for (int tokenId = 0; tokenId < vocabSize; tokenId++) {
            String word = tokenizer.getToken(tokenId); // Supposons que cette méthode existe
            if (wordVectors.hasWord(word)) {
                INDArray wordVector = wordVectors.getWordVectorMatrix(word);
                embeddings.putRow(tokenId, wordVector);
            } else {
                // Utiliser le vecteur moyen pour les mots inconnus
                embeddings.putRow(tokenId, meanVector);
            }
        }

        return embeddings;
    }

    public void train(DataGenerator dataGenerator) throws IOException {
        for (int epoch = 0; epoch < 10; epoch++) {
            int batchNum = 0;
            
            optimizer.setEpoch(epoch);
            
            // Itération sur les batches générés par le générateur de données
            while (dataGenerator.hasNextBatch()) {
                Batch batch = dataGenerator.nextBatch();
                
                // Tokenization du texte cible et conversion en IDs
                List<Integer> targetTokenIds = tokenizer.tokensToIds(tokenizer.tokenize(String.join("", batch.getTarget())));
                
                // Tokenization du texte source et conversion en IDs
                List<Integer> dataTokenIds = tokenizer.tokensToIds(tokenizer.tokenize(String.join("", batch.getData())));

                // Encodage et décodage pour obtenir les logits
                INDArray encoded = encoder.encode(dataTokenIds);
                List<INDArray> decodedLogits = decoder.decode(encoded); // Assumer que decode retourne désormais List<INDArray>
                            
                // Calcul de la perte
                float loss = calculateCrossEntropyLoss(decodedLogits, targetTokenIds);

                // Mise à jour des paramètres via l'optimiseur
                List<INDArray> combinedParameters = getCombinedParameters();
                List<INDArray> combinedGradients = getCombinedGradients(loss);
                optimizer.update(combinedParameters, combinedGradients);
                
                batchNum++;
            }
            
            dataGenerator.init(); // Réinitialiser le générateur de données si nécessaire
        }

        isTrained = true;
    }

    
    private List<INDArray> getCombinedParameters() {
        List<INDArray> combinedParameters = new ArrayList<>();
        
        // Ajoute les paramètres de l'encoder
        combinedParameters.addAll(encoder.getParameters());
        
        // Ajoute les paramètres du decoder
        combinedParameters.addAll(decoder.getParameters());
        
        return combinedParameters;
    }

    private List<INDArray> getCombinedGradients(float loss) {
        List<INDArray> combinedGradients = new ArrayList<INDArray>();
        
        INDArray encoderGradients = encoder.calculateGradients(loss);
        INDArray decoderGradients = decoder.calculateGradients(loss);
        
        // Supposons que chaque composant a une méthode pour obtenir ses gradients après backpropagation
        combinedGradients.add(encoderGradients);
        combinedGradients.add(decoderGradients);
        
        return combinedGradients;
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

    private float calculateCrossEntropyLoss(List<INDArray> decodedLogits, List<Integer> targetTokenIds) {
        float loss = 0.0f;
        int N = targetTokenIds.size();

        // Assumons que decodedLogits contient une seule INDArray pour l'ensemble de la séquence
        INDArray logits = decodedLogits.get(0); // Obtenez les logits pour l'ensemble de la séquence

        for (int i = 0; i < N; i++) {
            int targetId = targetTokenIds.get(i); // L'ID attendu à la position i

            // Extraire les logits pour la position i et toutes les classes (vocabulaire)
            INDArray logitsForPosition = logits.getRow(i); // Assume une forme [vocabSize] pour chaque position
            
            // Utiliser Transforms pour le softmax sur les logits pour la position i
            INDArray softmaxLogits = Transforms.softmax(logitsForPosition, false); 
            
            // Calculer le log softmax spécifiquement pour l'indice de la cible
            float logSoftmaxForTarget = (float) Math.log(softmaxLogits.getDouble(targetId));
            
            // Accumuler la perte négative log softmax pour la cible
            loss += -logSoftmaxForTarget;
        }

        return loss / N; // Moyenne de la perte sur la longueur de la séquence
    }



}
