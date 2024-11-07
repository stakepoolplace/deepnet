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
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.utils.NDArrayUtils;

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
    private int dModel = 300; // dmodel must be divisible by numHeads
    private int numLayers = 6;
    private int numHeads = 6; 
    private int dff = 2048;
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

    /**
     * Méthode par défaut pour le constructeur.
     *
     * @throws IOException en cas d'erreur de chargement des embeddings.
     */
    public TransformerModel() throws IOException {
        this(6, 300, 6, 2048, 0.1);
    }
    
    /**
     * Constructeur principal du modèle Transformer.
     *
     * @param numLayers    Nombre de couches dans l'encodeur et le décodeur.
     * @param dModel       Dimension du modèle.
     * @param numHeads     Nombre de têtes dans l'attention multi-têtes.
     * @param dff          Dimension de la couche Feed-Forward.
     * @param dropoutRate  Taux de dropout.
     */
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate) {

    	this.numLayers = numLayers;
    	this.dModel = dModel;
    	this.numHeads = numHeads;
    	this.dff = dff;
    	this.dropoutRate = dropoutRate;

        // Garantit la compatibilité et les performances optimales
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
    	
        // Charger Word2Vec
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(W2VECPATH));
        
        // Créer le tokenizer qui gère maintenant aussi les embeddings
        this.tokenizer = new Tokenizer(wordVectors);
        
        // Utiliser les embeddings du tokenizer
        pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate);   

        addCombinedParameters();
        addCombinedGradients();

        this.optimizer = new CustomAdamOptimizer(0.001, dModel, 1000, combinedParameters); // Initialisation hypothétique
    }


    public void train(DataGenerator dataGenerator) throws IOException {
        for (int epoch = 0; epoch < 10; epoch++) {
            optimizer.setEpoch(epoch);
            System.out.println("Starting epoch " + (epoch + 1));
    
            while (dataGenerator.hasNextBatch()) {
                
                // Nettoyer les gradients précédents
                cleanGradients();
                
                Batch batch = dataGenerator.nextBatch();
    
                // Tokeniser chaque phrase dans le batch séparément
                List<List<String>> tokenizedData = batch.getData().stream()
                    .map(sentence -> tokenizer.tokenize(sentence))
                    .collect(Collectors.toList());
    
                List<List<String>> tokenizedTargets = batch.getTarget().stream()
                    .map(sentence -> tokenizer.tokenize(sentence))
                    .collect(Collectors.toList());
    
                // Convertir les tokens en IDs pour chaque phrase du batch
                List<List<Integer>> dataTokenIds = tokenizedData.stream()
                    .map(tokens -> tokenizer.tokensToIds(tokens))
                    .collect(Collectors.toList());
    
                List<List<Integer>> targetTokenIds = tokenizedTargets.stream()
                    .map(tokens -> tokenizer.tokensToIds(tokens))
                    .collect(Collectors.toList());
    
                // Déterminer la longueur maximale dans le batch pour le padding
                int maxDataSeqLength = dataTokenIds.stream().mapToInt(List::size).max().orElse(0);
                int maxTargetSeqLength = targetTokenIds.stream().mapToInt(List::size).max().orElse(0);
    
                // Padder les séquences
                List<List<Integer>> paddedDataTokenIds = padSequences(dataTokenIds, maxDataSeqLength, tokenizer.getPadTokenId());
                List<List<Integer>> paddedTargetTokenIds = padSequences(targetTokenIds, maxTargetSeqLength, tokenizer.getPadTokenId());
    
                // Créer les masques de padding pour le batch
                INDArray encoderPaddingMask = createPaddingMask(paddedDataTokenIds);
                INDArray decoderPaddingMask = createPaddingMask(paddedTargetTokenIds);
                INDArray lookAheadMask = createLookAheadMask(maxTargetSeqLength);
    
                // Encoder les données du batch
                INDArray encoded = encoder.encode(true, paddedDataTokenIds, encoderPaddingMask);
                System.out.println("Encoded output shape: " + Arrays.toString(encoded.shape()));
    
                // Décoder les données encodées
                INDArray decodedOutput = decoder.decode(true, encoded, encoded, lookAheadMask, decoderPaddingMask);
                System.out.println("Decoded output shape: " + Arrays.toString(decodedOutput.shape()));
    
                // Calculer la perte et les gradients
                List<INDArray> decodedLogits = Arrays.asList(decodedOutput);
                backpropagation(decodedLogits, paddedTargetTokenIds);
    
                // Mettre à jour les paramètres du modèle via l'optimiseur
                optimizer.update(combinedParameters, combinedGradients);
    
                // (Optionnel) Afficher la progression
                System.out.println("Batch processed.");
            }
    
            // Réinitialiser le générateur de données pour le prochain epoch
            dataGenerator.init();
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }
    
        isTrained = true;
        System.out.println("Training completed.");
    }
    

    private List<List<Integer>> padSequences(List<List<Integer>> sequences, int maxLength, int padTokenId) {
        return sequences.stream()
            .map(seq -> {
                List<Integer> padded = new ArrayList<>(seq);
                while (padded.size() < maxLength) {
                    padded.add(padTokenId);
                }
                return padded;
            })
            .collect(Collectors.toList());
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
    

    
    public INDArray createPaddingMask(List<List<Integer>> tokenIdsBatch) {
        // Déterminer le batch size et la longueur maximale de séquence
        int batchSize = tokenIdsBatch.size();
        int seqLength = tokenIdsBatch.get(0).size(); // Supposons que toutes les séquences sont padées à seqLength
    
        // Créer un masque de zéros de la forme [batchSize, 1, 1, seqLength]
        INDArray mask = Nd4j.zeros(DataType.FLOAT, batchSize, 1, 1, seqLength);
    
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                if (tokenIdsBatch.get(i).get(j) == tokenizer.getPadTokenId()) { // Supposons que le token <PAD> a un ID spécifique
                    mask.putScalar(new int[]{i, 0, 0, j}, Float.NEGATIVE_INFINITY); // Utiliser -inf pour les positions à ignorer
                } else {
                    mask.putScalar(new int[]{i, 0, 0, j}, 0.0f); // Pas de masque pour les positions valides
                }
            }
        }
    
        return mask;
    }
    
    



    
    private void backpropagation(List<INDArray> decodedLogits, List<List<Integer>> targetTokenIdsBatch) {
        // Étape 1: Calcul de la perte et des gradients initiaux
        Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, targetTokenIdsBatch);
        float loss = lossAndGradients.getLeft();
        INDArray initialGradients = lossAndGradients.getRight();
    
        // Afficher la perte pour le monitoring
        System.out.println("Perte: " + loss);
    
        // Étape 2: Rétropropagation à travers le Décodeur
        Map<String, INDArray> decoderGradients = decoder.backward(initialGradients);
    
        // Extraire les gradients pertinents pour l'encodeur à partir de decoderGradients
        Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);
    
        // Étape 3: Rétropropagation à travers l'Encodeur
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
        // Récupérer les gradients de toutes les couches
        combinedGradients = new ArrayList<>();
        combinedGradients.addAll(encoder.getGradients());
        combinedGradients.addAll(decoder.getGradients());
    
        // Mettre à jour les poids du modèle via l'optimiseur
        optimizer.update(combinedParameters, combinedGradients);
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




    public String infer(String prompt, int maxLength) {
        if (!isTrained) {
            throw new IllegalStateException("Le modèle doit être entraîné avant l'inférence.");
        }
    
        // Tokenisation du prompt
        List<String> promptTokens = tokenizer.tokenize(prompt);
        List<Integer> promptTokenIds = tokenizer.tokensToIds(promptTokens);
    
        // Ajout du token de début si nécessaire
        promptTokenIds.add(0, tokenizer.getStartTokenId());
    
        // Déterminer la longueur maximale de séquence après ajout du token de début
        int maxSeqLength = promptTokenIds.size();
    
        // Padder la séquence si nécessaire
        List<Integer> paddedPromptTokenIds = padSequencesSingle(promptTokenIds, maxSeqLength, tokenizer.getPadTokenId());
    
        // Créer le masque de padding pour l'encodeur
        INDArray encoderPaddingMask = createPaddingMaskSingle(paddedPromptTokenIds);
    
        // Encoder le prompt (traité comme un batch de taille 1)
        INDArray encodedPrompt = encoder.encode(false, Arrays.asList(paddedPromptTokenIds), encoderPaddingMask);
        System.out.println("Encoded prompt shape: " + Arrays.toString(encodedPrompt.shape()));
    
        // Initialiser les IDs de sortie avec le token de début
        List<Integer> outputIds = new ArrayList<>();
        outputIds.add(tokenizer.getStartTokenId());
    
        for (int i = 0; i < maxLength; i++) {
            // Créer les masques pour le décodeur
            INDArray decoderPaddingMask = createPaddingMaskSingle(outputIds);
            INDArray lookAheadMask = createLookAheadMask(outputIds.size());
    
            // Convertir les IDs de sortie en embeddings pour le décodeur
            INDArray encodedDecoderInput = encoder.lookupEmbeddings(Arrays.asList(outputIds));
            System.out.println("Encoded decoder input shape: " + Arrays.toString(encodedDecoderInput.shape()));
    
            // Décoder
            INDArray logits = decoder.decode(false, encodedPrompt, encodedDecoderInput, lookAheadMask, decoderPaddingMask);
            System.out.println("Logits shape: " + Arrays.toString(logits.shape()));
    
            // Extraction des logits du dernier token généré
            // Correction de l'utilisation de INDArrayIndex
            int lastPosition = (int) logits.shape()[1] - 1; // seqLength - 1
            INDArray lastTokenLogits = logits.get(
                NDArrayIndex.point(0),                     // batch 0
                NDArrayIndex.point(lastPosition),          // dernière position dans seqLength
                NDArrayIndex.all()                         // tous les éléments dans vocabSize
            ).dup(); // [vocabSize]
    
            // Appliquer softmax pour obtenir les probabilités
            INDArray softmaxLogits = Transforms.softmax(lastTokenLogits, false);
    
            // Sélectionner le token avec la plus haute probabilité
            int predictedTokenId = Nd4j.argMax(softmaxLogits, 0).getInt(0);
    
            // Ajouter le token prédit à la séquence de sortie
            outputIds.add(predictedTokenId);
    
            // Vérification du token de fin
            if (predictedTokenId == tokenizer.getEndTokenId()) {
                break;
            }
    
            // (Optionnel) Implémentation d'une stratégie de décodage plus sophistiquée
            // Par exemple, beam search, échantillonnage avec température, etc.
            // Ceci nécessiterait une refonte plus approfondie de la boucle de génération.
        }
    
        // Conversion des IDs en tokens
        // Exclure le token de début
        return tokenizer.idsToTokens(outputIds.subList(1, outputIds.size()));
    }
    
    
    /**
     * Pad une seule séquence à une longueur maximale avec le token de padding.
     *
     * @param sequence      La séquence d'IDs de tokens.
     * @param maxLength     La longueur maximale à atteindre.
     * @param padTokenId    L'ID du token de padding.
     * @return La séquence padée.
     */
    private List<Integer> padSequencesSingle(List<Integer> sequence, int maxLength, int padTokenId) {
        List<Integer> padded = new ArrayList<>(sequence);
        while (padded.size() < maxLength) {
            padded.add(padTokenId);
        }
        return padded;
    }
    
    /**
     * Crée un masque de padding pour une seule séquence.
     *
     * @param tokenIds La liste d'IDs de tokens.
     * @return Un INDArray représentant le masque de padding.
     */
    public INDArray createPaddingMaskSingle(List<Integer> tokenIds) {
        int seqLength = tokenIds.size();
        // Créer un masque de zéros de la forme [1, 1, 1, seqLength]
        INDArray mask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, seqLength);
    
        for (int j = 0; j < seqLength; j++) {
            if (tokenIds.get(j) == tokenizer.getPadTokenId()) {
                mask.putScalar(new int[]{0, 0, 0, j}, Float.NEGATIVE_INFINITY);
            } else {
                mask.putScalar(new int[]{0, 0, 0, j}, 0.0f);
            }
        }
    
        return mask;
    }
    


    public boolean isTrained() {
        return isTrained;
    }


    protected Pair<Float, INDArray> calculateCrossEntropyLossAndGradient(List<INDArray> decodedLogits, List<List<Integer>> targetTokenIdsBatch) {
        float loss = 0.0f;
        int batchSize = targetTokenIdsBatch.size();
        int maxSeqLength = targetTokenIdsBatch.stream().mapToInt(List::size).max().orElse(0);
    
        // Assumons que decodedLogits contient une seule INDArray pour tout le batch
        INDArray logits = decodedLogits.get(0); // [batchSize, targetSeqLength, vocabSize]
        INDArray gradients = Nd4j.zeros(DataType.FLOAT, logits.shape()); // [batchSize, targetSeqLength, vocabSize]
    
        for (int b = 0; b < batchSize; b++) {
            List<Integer> targetTokenIds = targetTokenIdsBatch.get(b);
            for (int i = 0; i < targetTokenIds.size(); i++) {
                int targetId = targetTokenIds.get(i); // L'ID attendu à la position i
    
                // Extraire les logits pour la position i de la séquence b
                INDArray logitsForPosition = logits.get(
                    NDArrayIndex.point(b), 
                    NDArrayIndex.point(i), 
                    NDArrayIndex.all()
                ).dup(); // [vocabSize]
    
                // Appliquer softmax
                INDArray softmaxLogits = Transforms.softmax(logitsForPosition, false); 
    
                // Calculer le log softmax pour l'ID cible
                double prob = softmaxLogits.getDouble(targetId);
                // Éviter les problèmes avec log(0) en ajoutant une petite constante epsilon
                float logSoftmaxForTarget = (float) Math.log(Math.max(prob, 1e-10));
    
                // Accumuler la perte négative log softmax
                loss += -logSoftmaxForTarget;
    
                // Calculer le gradient (p - y)
                INDArray targetOneHot = Nd4j.zeros(DataType.FLOAT, softmaxLogits.shape());
                targetOneHot.putScalar(targetId, 1.0f);
                INDArray gradForPosition = softmaxLogits.sub(targetOneHot); // [vocabSize]
    
                // Mise à jour du gradient pour toute la tranche [b, i, :]
                gradients.put(
                    new INDArrayIndex[]{
                        NDArrayIndex.point(b), 
                        NDArrayIndex.point(i), 
                        NDArrayIndex.all()
                    },
                    gradForPosition
                );

            }
        }
    
        // Moyenne de la perte sur le batch
        loss /= batchSize;
    
        // Moyenne des gradients
        gradients.divi(batchSize);
    
        return Pair.of(loss, gradients); // Moyenne de la perte et gradients accumulés
    }
    
    
    
    public void saveState(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            
            this.writeObject(oos);
            
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
        // oos.defaultWriteObject();
        // Vous pouvez sauvegarder le chemin du fichier Word2Vec si nécessaire
        oos.writeObject(W2VECPATH);
    }

    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        // ois.defaultReadObject();
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


    /**
     * Initialise les embeddings pré-entraînés.
     *
     * @param vocabSize Taille du vocabulaire
     * @param dModel    Dimension du modèle
     */
    public static void initializeEmbeddings(int vocabSize, int dModel) {
        // Exemple : initialisation aléatoire des embeddings avec normalisation
        pretrainedEmbeddings = Nd4j.randn(DataType.FLOAT, vocabSize, dModel).divi(Math.sqrt(dModel));
    }

    /**
     * Récupère les embeddings pré-entraînés.
     *
     * @return INDArray contenant les embeddings [vocabSize, dModel]
     */
    public static INDArray getPretrainedEmbeddings() {
        if (pretrainedEmbeddings == null) {
            throw new IllegalStateException("Les embeddings pré-entraînés ne sont pas initialisés. Appelez initializeEmbeddings() d'abord.");
        }
        return pretrainedEmbeddings;
    }


	public List<INDArray> getCombinedParameters() {
		return combinedParameters;
	}


	public List<INDArray> getCombinedGradients() {
		return combinedGradients;
	}







}
