package RN.transformer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TransformerModel  implements Serializable {
	
    /**
	 * 
	 */
	private static final long serialVersionUID = -4799769434788429831L;
	
	private static String W2VECPATH = "pretrained-embeddings/mon_model_word2vec.txt";
	private boolean isTrained = false;
    public Encoder encoder;
    public Decoder decoder;
    public CustomAdamOptimizer optimizer;
    public Tokenizer tokenizer;
    private double dropoutRate = 0.1; // Exemple de taux de dropout fixe
    private transient static WordVectors wordVectors; // Chargé une fois, accessible statiquement
    private static int dModel = 300; // dmodel must be divisible by numHeads
    private static int numLayers = 6;
    private static int numHeads = 6; 
    private static int dff = 2048;
    private static INDArray pretrainedEmbeddings = null;
    private List<INDArray> combinedParameters = new ArrayList<>();
    private List<INDArray> combinedGradients = new ArrayList<INDArray>();
    
    static {
        try {
        	wordVectors = WordVectorSerializer.readWord2VecModel(new File(W2VECPATH), true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate) {
    	this.numLayers = numLayers;
    	this.dModel = dModel;
    	this.numHeads = numHeads;
    	this.dff = dff;
    	this.dropoutRate = dropoutRate;
    	
        // Charger Word2Vec
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(W2VECPATH));
        
        // Créer le tokenizer qui gère maintenant aussi les embeddings
        this.tokenizer = new Tokenizer(wordVectors);
        
        // Utiliser les embeddings du tokenizer
        pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate);
        
        // Calcul du nombre total de paramètres
        long totalParams = encoder.getNumberOfParameters() + decoder.getNumberOfParameters();
        
        this.optimizer = new CustomAdamOptimizer(0.001, dModel, 1000, totalParams); // Initialisation hypothétique
    }
    

    public TransformerModel() throws IOException {
        
        // Charger Word2Vec
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(W2VECPATH));
        
        // Créer le tokenizer qui gère maintenant aussi les embeddings
        this.tokenizer = new Tokenizer(wordVectors);
        
        // Utiliser les embeddings du tokenizer
        pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate);
        
        // Calcul du nombre total de paramètres
        long totalParams = encoder.getNumberOfParameters() + decoder.getNumberOfParameters();
        
        this.optimizer = new CustomAdamOptimizer(0.001, dModel, 1000, totalParams); // Initialisation hypothétique
    }
    
 
    public void train(DataGenerator dataGenerator) throws IOException {
        for (int epoch = 0; epoch < 10; epoch++) {
            optimizer.setEpoch(epoch);

            while (dataGenerator.hasNextBatch()) {
            	
            	// at start because it keeps gradients loaded for the last loop and we can save the state on the disk if needed
            	cleanGradients();
            	
                Batch batch = dataGenerator.nextBatch();

                List<Integer> targetTokenIds = tokenizer.tokensToIds(tokenizer.tokenize(String.join("", batch.getTarget())));
                List<Integer> dataTokenIds = tokenizer.tokensToIds(tokenizer.tokenize(String.join("", batch.getData())));

                // Créer les masques
                INDArray encoderPaddingMask = createPaddingMask(dataTokenIds);
                INDArray decoderPaddingMask = createPaddingMask(targetTokenIds);
                INDArray lookAheadMask = createLookAheadMask(targetTokenIds.size());

                INDArray encoded = encoder.encode(true, dataTokenIds, encoderPaddingMask);
                INDArray decodedOutput = decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);

                List<INDArray> decodedLogits = new ArrayList<>();
                decodedLogits.add(decodedOutput);

                backpropagation(decodedLogits, targetTokenIds);
                addCombinedParameters();
                addCombinedGradients();
                optimizer.update(combinedParameters, combinedGradients);
            }

            dataGenerator.init();
        }

        isTrained = true;
    }
    
    public INDArray createLookAheadMask(int size) {
        // Création d'une matrice où les éléments au-dessus de la diagonale sont 1 (ce qui signifie masqués)
        INDArray mask = Nd4j.ones(size, size);
        INDArray lowerTriangle = Nd4j.tri(size, size, 0); // Crée une matrice triangulaire inférieure
        mask.subi(lowerTriangle); 
        // Appliquer dessous le masquage infini pour softmax
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                mask.putScalar(i, j, Double.NEGATIVE_INFINITY);
            }
        }        
        return mask;
    }
    

    
    public INDArray createPaddingMask(List<Integer> tokenIds) {
        // Génération d'un masque où chaque emplacement de padding est marqué par 1 (infinité après le masquage)
        long size = tokenIds.size();
        INDArray mask = Nd4j.zeros(size);
        for (int i = 0; i < size; i++) {
            if (tokenIds.get(i) == tokenizer.getPadTokenId()) { 
                mask.putScalar(i, Double.POSITIVE_INFINITY);
            }
        }
        return mask;
    }
    



    
    private void backpropagation(List<INDArray> decodedLogits, List<Integer> targetTokenIds) {
        // Étape 1: Calcul de la perte et des gradients initiaux
        // Cette fonction est hypothétique et devrait retourner la perte et le gradient initial
        Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, targetTokenIds);
        float loss = lossAndGradients.getLeft();
        INDArray initialGradients = lossAndGradients.getRight();
        
        // Afficher la perte pour le monitoring
        System.out.println("Perte: " + loss);

        // Étape 2: Rétropropagation à travers le Décodeur
        // Cela ajuste les poids du décodeur basés sur les gradients calculés
        Map<String, INDArray> decoderGradients = decoder.backward(initialGradients);
        
        // Extraire les gradients pertinents pour l'encodeur à partir de decoderGradients
        Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);
        

        // Étape 3: Rétropropagation à travers l'Encodeur
        // L'encodeur ajuste ses poids basé sur ses propres calculs de gradients
        // Dans un modèle Transformer, cela pourrait impliquer des gradients venant de la couche d'attention encodeur-décodeur
        // Pour simplifier, nous allons juste appeler backward sur l'encodeur sans passer de gradients spécifiques
        // car dans une implémentation réelle, cela dépendrait des détails spécifiques de votre modèle
        encoder.backward(encoderGradients);

        // Mettre à jour les poids basés sur les gradients calculés, normalement fait par l'optimiseur
        updateModelWeights();
    }


    private Map<String, INDArray> extractEncoderGradients(Map<String, INDArray> decoderGradients) {
        // Créez un nouveau Map pour contenir les gradients spécifiquement pour l'encoder.
        Map<String, INDArray> encoderGradients = new HashMap<>();
        
        // Extrayez les gradients par rapport aux entrées K et V de l'attention encoder-décodeur.
        // Ces gradients sont ceux qui doivent être propagés à travers l'encoder.
        INDArray gradK = decoderGradients.get("inputK");
        INDArray gradV = decoderGradients.get("inputV");
        
        // Ajoutez ces gradients au Map sous des clés représentant leur rôle dans l'encoder.
        // Par exemple, vous pouvez simplement les renommer pour correspondre à la nomenclature attendue par l'encoder.
        encoderGradients.put("gradK", gradK);
        encoderGradients.put("gradV", gradV);
        
        return encoderGradients;
    }




	private void updateModelWeights() {
        // Implémentez cette fonction pour mettre à jour les poids du modèle
        // basé sur les gradients calculés. Normalement, cela est géré par votre optimiseur
    }



	public void addCombinedParameters() {
        
        // Ajoute les paramètres de l'encoder
        combinedParameters.addAll(encoder.getParameters());
        
        // Ajoute les paramètres du decoder
        combinedParameters.addAll(decoder.getParameters());
        
    }

    private void addCombinedGradients() {
        
        // Ajoute les gradients de l'encoder
        combinedGradients.addAll(encoder.getGradients());
        
        // Ajoute les gradients du decoder
        combinedGradients.addAll(decoder.getGradients());
        
    }
    
    public void cleanGradients() {
    	combinedParameters.clear();
    	combinedGradients.clear();
    }




    public String infer(String prompt) {
        if (!isTrained) {
            throw new IllegalStateException("Le modèle doit être entraîné avant l'inférence.");
        }

        List<String> promptTokens = tokenizer.tokenize(prompt);
        List<Integer> promptTokenIds = tokenizer.tokensToIds(promptTokens);

        INDArray encoderPaddingMask = createPaddingMask(promptTokenIds);
        INDArray encodedPrompt = encoder.encode(false, promptTokenIds, encoderPaddingMask);

        List<Integer> outputIds = new ArrayList<>();
        int maxLength = 100; // Définissez une longueur maximale pour la sortie

        for (int i = 0; i < maxLength; i++) {
            List<Integer> currentOutput = new ArrayList<>(promptTokenIds);
            currentOutput.addAll(outputIds);

            INDArray decoderPaddingMask = createPaddingMask(currentOutput);
            INDArray lookAheadMask = createLookAheadMask(currentOutput.size());

            INDArray logits = decoder.decode(false, encodedPrompt, encodedPrompt, lookAheadMask, decoderPaddingMask);

            // Prendre le dernier token prédit
            INDArray lastTokenLogits = logits.get(NDArrayIndex.point(logits.rows() - 1), NDArrayIndex.all());
            int predictedTokenId = Nd4j.argMax(lastTokenLogits).getInt(0);

            outputIds.add(predictedTokenId);

            if (predictedTokenId == tokenizer.getEndTokenId()) {
                break;
            }
        }

        return tokenizer.idsToTokens(outputIds);
    }






    public boolean isTrained() {
        return isTrained;
    }


    
    protected Pair<Float, INDArray> calculateCrossEntropyLossAndGradient(List<INDArray> decodedLogits, List<Integer> targetTokenIds) {
        float loss = 0.0f;
        int N = targetTokenIds.size();
        

        // Assumons que decodedLogits contient une seule INDArray pour l'ensemble de la séquence
        INDArray logits = decodedLogits.get(0); // Obtenez les logits pour l'ensemble de la séquence
        INDArray gradients = Nd4j.zeros(logits.shape()); // Initialiser le gradient de la même forme que les logits

        System.out.println("logits.shape()"  + Arrays.toString(logits.shape()));
        
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

            // Calcul du gradient initial : p - y
            INDArray targetOneHot = Nd4j.zeros(logitsForPosition.shape());
            targetOneHot.putScalar(targetId, 1);
            INDArray gradForPosition = softmaxLogits.sub(targetOneHot);
            gradients.putRow(i, gradForPosition);
        }
        
        return Pair.of(loss / N, gradients); // Retourner la moyenne de la perte et les gradients accumulés
    }
    
    
    public void saveState(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            // Sauvegarder l'état de l'encodeur et du décodeur
            oos.writeObject(encoder);
            oos.writeObject(decoder);
            
            // Sauvegarder l'état de l'optimiseur
            oos.writeObject(optimizer.getCurrentStep());
            oos.writeObject(optimizer.getEpoch());
            oos.writeObject(optimizer.getLearningRate());
            
            // Sauvegarder les paramètres du modèle
            oos.writeObject(combinedParameters.size());
            for (INDArray param : combinedParameters) {
                oos.writeObject(param);
            }
            
            // Sauvegarder l'état d'entraînement
            oos.writeBoolean(isTrained);
        }
    }

    public void loadState(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
        	
        	this.readObject(ois);
        	
            // Charger l'état de l'encodeur et du décodeur
            this.encoder = (Encoder) ois.readObject();
            this.decoder = (Decoder) ois.readObject();
            
            // Charger l'état de l'optimiseur
            int currentStep = (int) ois.readObject();
            int epoch = (int) ois.readObject();
            double learningRate = (double) ois.readObject();
            optimizer.setCurrentStep(currentStep);
            optimizer.setEpoch(epoch);
            optimizer.setLearningRate(learningRate);
            
            // Charger les paramètres du modèle
            int numParams = (int) ois.readObject();
            for (int i = 0; i < numParams; i++) {
                INDArray param = (INDArray) ois.readObject();
                combinedParameters.get(i).assign(param);
            }
            
            // Charger l'état d'entraînement
            this.isTrained = ois.readBoolean();
        }
    }

    
    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject();
        // Vous pouvez sauvegarder le chemin du fichier Word2Vec si nécessaire
        oos.writeObject(W2VECPATH);
    }

    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        ois.defaultReadObject();
        String word2vecPath = (String) ois.readObject();
        // Réinitialiser wordVectors
        this.wordVectors = WordVectorSerializer.loadStaticModel(new File(word2vecPath));
    }



	public void setTrained(boolean isTrained) {
		this.isTrained = isTrained;
	}



	public int getDModel() {
		return dModel;
	}



	public static int getVocabSize() {
		return pretrainedEmbeddings.rows();
	}


	public static INDArray getPretrainedEmbeddings() {
		return pretrainedEmbeddings;
	}


	public List<INDArray> getCombinedParameters() {
		return combinedParameters;
	}


	public List<INDArray> getCombinedGradients() {
		return combinedGradients;
	}


}
