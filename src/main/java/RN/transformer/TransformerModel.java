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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.transformer.Decoder.DecoderLayer;
import RN.transformer.Encoder.EncoderLayer;
import RN.utils.NDArrayUtils;

public class TransformerModel implements Serializable {

    private static final long serialVersionUID = -4799769434788429831L;

    // Affiche les tableaux d'attention croisée (encoder : self  decoder: self, cross)
    private final static boolean TRACE_ON = true;

    private static final String W2VECPATH = "pretrained-embeddings/mon_model_word2vec.txt";
    private boolean isTrained = false;
    public Encoder encoder;
    public Decoder decoder;
    public CustomAdamOptimizer optimizer;
    public Tokenizer tokenizer;
    private double dropoutRate = 0.01; // Exemple de taux de dropout fixe
    private transient static WordVectors wordVectors; // Chargé une fois, accessible statiquement
    private int dModel = 300; // dModel must be divisible by numHeads
    private int numLayers = 6;
    private int numHeads = 6;
    private int dff = 2048;
    private int maxSequenceLength = 50; // Définissez cette valeur selon votre analyse
    private INDArray pretrainedEmbeddings = null; // Maintenant non statique
    private List<INDArray> combinedParameters = new ArrayList<>();
    private List<INDArray> combinedGradients = new ArrayList<>();
    

    static {
        try {
            wordVectors = WordVectorSerializer.readWord2VecModel(new File(W2VECPATH), true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Constructeur par défaut du modèle Transformer.
     *
     * @throws IOException en cas d'erreur de chargement des embeddings.
     */
    public TransformerModel(float initialLr, int warmupSteps) throws IOException {

        // Garantit la compatibilité et les performances optimales
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        // Définir un vocabulaire par défaut si nécessaire
        List<String> defaultVocab = Arrays.asList("hello", "world", "test", "input", "output", "The", "quick", "brown",
                "fox", "jumps", "over", "the", "lazy", "dog", "<PAD>", "<UNK>", "<START>", "<END>");
        this.tokenizer = new Tokenizer(defaultVocab, dModel, maxSequenceLength);
        this.pretrainedEmbeddings = this.tokenizer.getPretrainedEmbeddings();

        // Initialiser les autres composants
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);

        addCombinedParameters();

        this.optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, combinedParameters); // Initialisation
                                                                                                      // hypothétique

        freezeSpecialTokenEmbeddings();
    }

    /**
     * Constructeur principal du modèle Transformer.
     *
     * @param numLayers   Nombre de couches dans l'encodeur et le décodeur.
     * @param dModel      Dimension du modèle.
     * @param numHeads    Nombre de têtes dans l'attention multi-têtes.
     * @param dff         Dimension de la couche Feed-Forward.
     * @param dropoutRate Taux de dropout.
     */
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, float initialLr,
            int warmupSteps) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dff = dff;
        this.dropoutRate = dropoutRate;

        // Garantit la compatibilité et les performances optimales
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        // Initialiser le Tokenizer avec WordVectors
        if (wordVectors != null) {
            this.tokenizer = new Tokenizer(wordVectors, dModel, maxSequenceLength);
            this.pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();
        } else {
            // Si les WordVectors ne sont pas chargés, initialiser avec un vocabulaire vide
            // ou par défaut
            List<String> defaultVocab = Arrays.asList("<PAD>", "<UNK>", "<START>", "<END>");
            this.tokenizer = new Tokenizer(defaultVocab, dModel, maxSequenceLength);
            this.pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();
        }

        // Initialiser l'encodeur et le décodeur avec le tokenizer
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);

        addCombinedParameters();

        // Initialiser l'optimiseur avec les paramètres combinés
        this.optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, combinedParameters);

        freezeSpecialTokenEmbeddings();
    }

    /**
     * Nouveau constructeur compatible avec le test.
     *
     * @param numLayers   Nombre de couches dans l'encodeur et le décodeur.
     * @param dModel      Dimension du modèle.
     * @param numHeads    Nombre de têtes dans l'attention multi-têtes.
     * @param dff         Dimension de la couche Feed-Forward.
     * @param dropoutRate Taux de dropout.
     * @param vocabSize   Taille du vocabulaire.
     * @param tokenizer   Instance de Tokenizer personnalisée.
     */
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, int vocabSize,
            Tokenizer tokenizer, float initialLr, int warmupSteps) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dff = dff;
        this.dropoutRate = dropoutRate;
        this.tokenizer = tokenizer;

        // Garantit la compatibilité et les performances optimales
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        // Utiliser les embeddings du tokenizer
        this.pretrainedEmbeddings = tokenizer.getPretrainedEmbeddings();

        // Si les embeddings sont null, initialisez-les avec des embeddings aléatoires
        if (this.pretrainedEmbeddings == null) {
            System.out.println("Pretrained embeddings are null. Initializing random embeddings.");
            this.pretrainedEmbeddings = Nd4j.randn(DataType.FLOAT, vocabSize, dModel).divi(Math.sqrt(dModel));
            tokenizer.setPretrainedEmbeddings(this.pretrainedEmbeddings); // Assurez-vous que Tokenizer peut définir les
                                                                          // embeddings
        }

        // Initialiser l'encodeur et le décodeur avec le tokenizer personnalisé
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);

        addCombinedParameters();

        // Initialiser l'optimiseur avec les paramètres combinés
        this.optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, combinedParameters);

        freezeSpecialTokenEmbeddings();
    }

    // Méthode pour initialiser l'optimiseur après avoir collecté tous les
    // paramètres
    public void initializeOptimizer(float learningRate, int warmupSteps) {
        List<INDArray> combinedParameters = this.encoder.getParameters();
        combinedParameters.addAll(this.decoder.getParameters());
        this.optimizer = new CustomAdamOptimizer(learningRate, dModel, warmupSteps, combinedParameters);
    }

    /**
     * Rétropropagation des gradients à travers l'encodeur et le décodeur.
     *
     * @param gradOutput Les gradients de la perte par rapport aux sorties du
     *                   décodeur.
     */
    public void backward(INDArray gradOutput) {
        // Rétropropagation à travers le décodeur
        Map<String, INDArray> decoderGradients = decoder.backward(gradOutput);

        // Extraire les gradients pertinents pour l'encodeur à partir de
        // decoderGradients
        Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);

        // Rétropropagation à travers l'encodeur
        Map<String, INDArray> encoderBackwardGradients = encoder.backward(encoderGradients);

        // Récupérer 'gradEmbeddings' et le passer au Tokenizer
        INDArray gradEmbeddings = encoderBackwardGradients.get("gradEmbeddings");
        if (gradEmbeddings == null) {
            throw new IllegalStateException("Les gradients des embeddings sont absents.");
        }

        // Rétropropagation à travers le tokenizer (embeddings)
        tokenizer.backward(gradEmbeddings); // gradEmbeddings doit être de forme [batchSize, seqLength, dModel]
    }

    /**
     * Entraîne le modèle sur un nombre donné d'epochs.
     *
     * @param dataGenerator Générateur de données pour l'entraînement.
     * @param epochNum      Nombre d'epochs à entraîner.
     * @throws IOException En cas d'erreur d'E/S.
     */
    public void train(DataGenerator dataGenerator, int epochNum) throws IOException {

        for (int epoch = 0; epoch < epochNum; epoch++) {
            // Définir le numéro d'epoch actuel (commence à 1)
            optimizer.setEpoch(epoch + 1);
            System.out.println("Epoch " + (epoch + 1) + " / " + epochNum);

            float totalLoss = 0.0f;
            int totalTokens = 0;

            while (dataGenerator.hasNextBatch()) {
                // Nettoyer les gradients précédents
                cleanGradients();

                // Obtenir le prochain batch
                Batch batch = dataGenerator.nextBatch();
                INDArray input = batch.getData(); // [batchSize, seqLength]
                INDArray target = batch.getTarget(); // [batchSize, seqLength]
                // INDArray mask = batch.getMask(); // [batchSize, 1, 1, seqLength] (Non utilisé
                // directement ici)

                // Encoder les données du batch
                INDArray encoded = encoder.encode(true, batch);

                // Décoder les données encodées
                INDArray decodedOutput = decoder.decode(true, encoded, encoded, batch, null);


                // Affiche les tableaux d'attention croisée (encoder : self  decoder: self, cross)
                if (TRACE_ON) {
                    traceAttentionInEncoderAndDecoder(input, target);
                }

                // Calculer la perte et les gradients
                List<INDArray> decodedLogits = Arrays.asList(decodedOutput);
                Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, target);
                float loss = lossAndGradients.getLeft();
                INDArray initialGradients = lossAndGradients.getRight();

                totalLoss += loss;
                totalTokens += target.sumNumber().intValue(); // Total de tokens cibles pour normaliser la perte

                // Afficher la perte pour ce batch
                System.out.println("Perte pour ce batch: " + loss);

                // Rétropropagation
                backward(initialGradients);

                // Collecter tous les gradients
                collectCombinedGradients();

                // Vérifier et loguer les gradients pour 'chat'
                // logGradientForToken("chat");

                // Clip gradients
                clipGradients(combinedGradients, 0.5f); // Exemple avec maxNorm = 0.5

                // Mettre à jour les poids du modèle via l'optimiseur
                optimizer.update(combinedParameters, combinedGradients);

                // Appliquer les gradients aux embeddings
                tokenizer.applyGradients(optimizer.getLearningRate());

            }

            // Calculer la perte moyenne pour l'epoch
            float averageLoss = totalLoss / totalTokens;
            System.out.println("Epoch " + (epoch + 1) + " completed with average loss: " + averageLoss);

            // Réinitialiser le générateur de données pour le prochain epoch
            dataGenerator.init();

            // Marquer le modèle comme entraîné après le premier epoch
            if (epoch + 1 >= 1) {
                isTrained = true;
            }
        }

        System.out.println("Training completed.");
    }

    private void traceAttentionInEncoderAndDecoder(INDArray input, INDArray target) {
        // Convertir les IDs en tokens pour chaque échantillon du batch
        List<List<String>> batchInputQueryTokens = new ArrayList<>();
        List<List<String>> batchInputKeyTokens = new ArrayList<>();
        List<List<String>> batchTargetQueryTokens = new ArrayList<>();

        for (int i = 0; i < input.size(0); i++) { // Pour chaque échantillon
            List<Integer> inputTokenIds = new ArrayList<>();
            for (int j = 0; j < input.size(1); j++) {
                inputTokenIds.add((int) input.getInt(i, j));
            }
            List<String> inputQueryTokens = tokenizer.idsToListTokens(inputTokenIds);
            List<String> inputKeyTokens = inputQueryTokens; // Pour le self-attention de l'encodeur

            batchInputQueryTokens.add(inputQueryTokens);
            batchInputKeyTokens.add(inputKeyTokens);
        }

        // Convertir les IDs cibles en tokens pour chaque échantillon du batch
        for (int i = 0; i < target.size(0); i++) { // Pour chaque échantillon
            List<Integer> targetTokenIds = new ArrayList<>();
            for (int j = 0; j < target.size(1); j++) {
                targetTokenIds.add((int) target.getInt(i, j));
            }
            List<String> targetQueryTokens = tokenizer.idsToListTokens(targetTokenIds);
            batchTargetQueryTokens.add(targetQueryTokens);
        }

        // Afficher les relations d'attention pour chaque échantillon du batch
        displayAttentionRelations(batchInputQueryTokens, batchInputKeyTokens, batchTargetQueryTokens);
    }

    public float trainEpoch(DataGenerator dataGenerator) throws IOException {
        float totalLoss = 0.0f;
        int totalTokens = 0;

        // Initialiser pour l'epoch
        optimizer.setEpoch(optimizer.getEpoch() + 1);
        System.out.println("Epoch " + optimizer.getEpoch());

        while (dataGenerator.hasNextBatch()) {

            // Nettoyer les gradients précédents
            cleanGradients();

            Batch batch = dataGenerator.nextBatch();
            INDArray input = batch.getData(); 
            INDArray target = batch.getTarget(); // [batchSize, seqLength]

            // Encoder les données du batch
            INDArray encoded = encoder.encode(true, batch);

            // Décoder les données encodées
            INDArray decodedOutput = decoder.decode(true, encoded, encoded, batch, null);

            // Affiche les tableaux d'attention croisée (encoder : self  decoder: self, cross)
            if (TRACE_ON) {
                traceAttentionInEncoderAndDecoder(input, target);
            }

            // Calculer la perte et les gradients
            List<INDArray> decodedLogits = Arrays.asList(decodedOutput);
            Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, target);
            float loss = lossAndGradients.getLeft();
            INDArray initialGradients = lossAndGradients.getRight();

            totalLoss += loss;
            totalTokens += target.sumNumber().intValue(); // Total de tokens cibles pour normaliser la perte

            // Afficher la perte pour le monitoring
            System.out.println("Perte pour ce batch: " + loss);

            // Backward pass
            backward(initialGradients);

            // Collecter tous les gradients
            collectCombinedGradients();

            // Vérifier et loguer les gradients pour 'chat'
            // logGradientForToken("chat");

            // Clip gradients
            clipGradients(combinedGradients, 0.5f); // Exemple avec maxNorm = 0.5

            // Mettre à jour les poids du modèle via l'optimiseur
            optimizer.update(combinedParameters, combinedGradients);

            // Appliquer les gradients aux embeddings
            tokenizer.applyGradients(optimizer.getLearningRate());
        }

        // Calculer la perte moyenne
        float averageLoss = totalLoss / totalTokens;

        // Réinitialiser le générateur de données pour le prochain epoch
        dataGenerator.init();
        System.out.println("Epoch " + optimizer.getEpoch() + " completed with average loss: " + averageLoss);

        // Marquer le modèle comme entraîné après le premier epoch
        if (optimizer.getEpoch() >= 1) {
            isTrained = true;
        }

        return averageLoss;
    }

    /**
     * Vérifie et logue les gradients pour un token spécifique.
     *
     * @param token Le token à vérifier (par exemple, "chat").
     */
    private void logGradientForToken(String token) {
        Integer tokenId = tokenizer.getTokenToId().get(token);
        // System.out.println("ID de '" + token + "': " + tokenId);
        if (tokenId == null) {
            System.out.println("Le token '" + token + "' n'existe pas dans le vocabulaire.");
            return;
        }

        // Trouver l'index des embeddings dans combinedParameters
        int embeddingsParamIndex = combinedParameters.indexOf(pretrainedEmbeddings);
        if (embeddingsParamIndex == -1) {
            throw new IllegalStateException("Les embeddings ne sont pas trouvés dans combinedParameters.");
        }

        // Récupérer le gradient correspondant
        INDArray embeddingsGrad = combinedGradients.get(embeddingsParamIndex); // [vocabSize, dModel]
        INDArray gradChat = embeddingsGrad.getRow(tokenId).dup(); // [dModel]
        System.out.println("Gradient pour '" + token + "' embedding: " + gradChat);

        // Vérifier si le gradient pour le token est non nul
        if (gradChat.sumNumber().doubleValue() == 0.0) {
            System.out.println("Gradient pour '" + token + "' est 0");
        }
    }

    private void clipGradients(List<INDArray> gradients, double maxNorm) {
        for (INDArray grad : gradients) {
            double norm = grad.norm2Number().doubleValue();
            if (norm > maxNorm) {
                grad.divi(norm).muli(maxNorm);
            }
        }
    }

    /**
     * Pad une liste de séquences à une longueur maximale avec le token de padding.
     *
     * @param sequences  Liste de séquences d'IDs.
     * @param maxLength  Longueur maximale à atteindre.
     * @param padTokenId ID du token de padding.
     * @return Liste de séquences padées.
     */
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

    /**
     * Calcule la perte d'entropie croisée et les gradients associés.
     *
     * @param decodedLogits Logits générés par le décodeur [batchSize, seqLength,
     *                      vocabSize].
     * @param targetBatch   INDArray contenant les IDs des tokens cibles [batchSize,
     *                      seqLength].
     * @return Un Pair contenant la perte moyenne et les gradients [batchSize,
     *         seqLength, vocabSize].
     */
    protected Pair<Float, INDArray> calculateCrossEntropyLossAndGradient(List<INDArray> decodedLogits,
            INDArray targetBatch) {

        INDArray logits = decodedLogits.get(0); // [batchSize, seqLength, vocabSize]
        int batchSize = (int) logits.shape()[0];
        int seqLength = (int) logits.shape()[1];
        int vocabSize = (int) logits.shape()[2];

        // Appliquer le softmax de manière stable
        INDArray probabilities = NDArrayUtils.stableSoftmax(logits, 2); // [batchSize, seqLength, vocabSize]
        // System.out.println("Probabilités Calculées: " + probabilities);

        // Vérifier les probabilités
        if (probabilities.isNaN().any() || probabilities.isInfinite().any()) {
            throw new RuntimeException("Probabilités contiennent des NaN ou des valeurs infinies.");
        }

        // Créer une INDArray one-hot pour les cibles
        INDArray targetOneHot = Nd4j.zeros(DataType.FLOAT, batchSize, seqLength, vocabSize);
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLength; i++) {
                int targetId = targetBatch.getInt(b, i);
                if (targetId >= 0 && targetId < vocabSize) {
                    targetOneHot.putScalar(new int[] { b, i, targetId }, 1.0f);
                } else {
                    throw new IllegalArgumentException("ID de cible invalide: " + targetId);
                }
            }
        }

        // System.out.println("targetOneHot: " + targetOneHot);

        // Créer un masque pour ignorer les `<PAD>`
        INDArray paddingMask = NDArrayUtils.createKeyPaddingMask(tokenizer, targetBatch); // [batchSize, 1, 1,
                                                                                          // seqLength]
        INDArray paddingMaskReshaped = paddingMask.reshape(batchSize, 1, seqLength); // [batchSize, 1, seqLength]

        // Calculer la perte d'entropie croisée en masquant les `<PAD>`
        INDArray logSoftmax = Transforms.log(probabilities.add(1e-10)); // éviter log(0)
        INDArray crossEntropy = logSoftmax.mul(targetOneHot).neg(); // [batchSize, seqLength, vocabSize]

        // Appliquer le masque binaire (1 pour valide, 0 pour padding)
        INDArray maskedCrossEntropy = crossEntropy
                .mul(paddingMaskReshaped.broadcast(batchSize, vocabSize, seqLength).permute(0, 2, 1)); // [batchSize,
                                                                                                       // seqLength,
                                                                                                       // vocabSize]

        // System.out.println("Cross Entropy:\n" + crossEntropy);
        // System.out.println("Masked Cross Entropy:\n" + maskedCrossEntropy);

        // Calculer la perte moyenne sur les tokens valides
        float loss = maskedCrossEntropy.sumNumber().floatValue()
                / targetBatch.neq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT).sumNumber().floatValue();

        // System.out.println("Total Loss: " + loss);

        // Calculer les gradients (softmax - targetOneHot) en masquant les `<PAD>`
        INDArray gradients = probabilities.sub(targetOneHot)
                .mul(paddingMaskReshaped.broadcast(batchSize, vocabSize, seqLength).permute(0, 2, 1))
                .div(targetBatch.neq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT).sumNumber().floatValue());

        // System.out.println("Gradients:\n" + gradients);

        // Vérifier les gradients
        if (gradients.isNaN().any() || gradients.isInfinite().any()) {
            throw new RuntimeException("Gradients contiennent des NaN ou des valeurs infinies.");
        }

        return Pair.of(loss, gradients);
    }

    /**
     * Rétropropagation des gradients à travers l'encodeur et le décodeur.
     *
     * @param decoderGradients Gradients retournés par le décodeur.
     * @return Map contenant les gradients spécifiques à l'encodeur.
     */
    private Map<String, INDArray> extractEncoderGradients(Map<String, INDArray> decoderGradients) {
        Map<String, INDArray> encoderGradients = new HashMap<>();

        // Extrayez les gradients pertinents
        INDArray gradAttentionOutputConcat = decoderGradients.get("gradAttentionOutputConcat");
        INDArray gradInputQ = decoderGradients.get("gradInputQ");
        INDArray gradInputK = decoderGradients.get("gradInputK");
        INDArray gradInputV = decoderGradients.get("gradInputV");

        if (gradAttentionOutputConcat == null || gradInputQ == null || gradInputK == null || gradInputV == null) {
            throw new IllegalStateException("Un ou plusieurs gradients nécessaires sont null.");
        }

        // Ajoutez les gradients au Map
        encoderGradients.put("gradAttentionOutputConcat", gradAttentionOutputConcat);
        encoderGradients.put("gradInputQ", gradInputQ);
        encoderGradients.put("gradInputK", gradInputK);
        encoderGradients.put("gradInputV", gradInputV);

        return encoderGradients;
    }

    /**
     * Ajoute les paramètres de l'encodeur et du décodeur à la liste combinée.
     */
    public void addCombinedParameters() {
        // Ajoute les paramètres de l'encodeur
        List<INDArray> encoderParams = encoder.getParameters();
        for (INDArray param : encoderParams) {
            if (!isSpecialTokenParameter(param)) {
                combinedParameters.add(param);
                // System.out.println("Paramètre ajouté à combinedParameters (Encodeur): " + param);
            } else {
                // System.out.println("Paramètre exclu (Encodeur, token spécial): " + param);
            }
        }

        // Ajoute les paramètres du decoder
        List<INDArray> decoderParams = decoder.getParameters();
        for (INDArray param : decoderParams) {
            if (!isSpecialTokenParameter(param)) {
                combinedParameters.add(param);
                // System.out.println("Paramètre ajouté à combinedParameters (Décodeur): " + param);
            } else {
                // System.out.println("Paramètre exclu (Décodeur, token spécial): " + param);
            }
        }

        // Ajoute les embeddings du tokenizer
        List<INDArray> tokenizerParams = tokenizer.getEmbeddings(); // Assurez-vous que cette méthode retourne la liste
                                                                    // correcte
        for (INDArray param : tokenizerParams) {
            if (!isSpecialTokenParameter(param)) {
                combinedParameters.add(param);
                // System.out.println("Paramètre ajouté à combinedParameters (Tokenizer): " + param);
            } else {
                // System.out.println("Paramètre exclu (Tokenizer, token spécial): " + param);
            }
        }

    }

    private boolean isSpecialTokenParameter(INDArray param) {
        boolean isSpecial = false;
        for (int tokenId : Arrays.asList(tokenizer.getPadTokenId(), tokenizer.getStartTokenId(),
                tokenizer.getEndTokenId(), tokenizer.getUnkTokenId())) {
            INDArray specialEmbedding = tokenizer.getPretrainedEmbeddings().getRow(tokenId);
            if (param == specialEmbedding) { // Vérifie la référence
                isSpecial = true;
                break;
            }
        }
        // System.out.println("Paramètre est un token spécial: " + isSpecial);
        return isSpecial;
    }

    /**
     * Gèle les embeddings des tokens spéciaux, en évitant de geler <UNK> s'il a des
     * valeurs calculées.
     */
    public void freezeSpecialTokenEmbeddings() {
        int[] specialTokenIds = { tokenizer.getPadTokenId(), tokenizer.getStartTokenId(), tokenizer.getEndTokenId(),
                tokenizer.getUnkTokenId() };

        for (int tokenId : specialTokenIds) {
            if (tokenId == tokenizer.getUnkTokenId()) {
                // Vérifier si <UNK> a des valeurs calculées
                if (!tokenizer.hasCalculatedEmbedding(tokenId)) {
                    INDArray embedding = tokenizer.getPretrainedEmbeddings().getRow(tokenId);
                    embedding.assign(Nd4j.zeros(dModel));
                    System.out.println("<UNK> embedding was frozen.");
                } else {
                    // Ne pas geler <UNK>
                    System.out.println("<UNK> embedding has calculated values. Skipping freezing.");
                }
            } else {
                // Geler les autres tokens spéciaux
                INDArray embedding = tokenizer.getPretrainedEmbeddings().getRow(tokenId);
                embedding.assign(Nd4j.zeros(dModel)); // Exemple pour <PAD>
                System.out.println("Token ID " + tokenId + " embedding was frozen.");
            }
        }
    }

    /**
     * Collecte tous les gradients des composants du modèle.
     */
    private void collectCombinedGradients() {
        combinedGradients.addAll(encoder.getGradients());
        combinedGradients.addAll(decoder.getGradients());
        combinedGradients.addAll(tokenizer.getGradients());
    }

    /**
     * Nettoie les gradients accumulés.
     */
    public void cleanGradients() {
        tokenizer.resetGradients();
        combinedGradients.clear();
    }

    /**
     * Met à jour les poids du modèle en utilisant les gradients combinés.
     */
    private void updateModelWeights() {
        // Vérifiez si les tailles correspondent avant de les passer à l'optimiseur
        if (combinedParameters.size() != combinedGradients.size()) {
            throw new IllegalArgumentException(
                    "La taille de la liste des paramètres et des gradients doit être la même.");
        }

        // Mettre à jour les poids du modèle via l'optimiseur
        optimizer.update(combinedParameters, combinedGradients);
    }

    /**
     * Sauvegarde l'état du modèle dans un fichier.
     *
     * @param filePath Chemin du fichier où sauvegarder l'état.
     * @throws IOException En cas d'erreur d'E/S.
     */
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

    /**
     * Charge l'état du modèle depuis un fichier.
     *
     * @param filePath Chemin du fichier à partir duquel charger l'état.
     * @throws IOException            En cas d'erreur d'E/S.
     * @throws ClassNotFoundException En cas de classe non trouvée lors de la
     *                                désérialisation.
     */
    public void loadState(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {

            this.readObject(ois);

            // Charger l'état de l'encodeur et du décodeur
            this.encoder = (Encoder) ois.readObject();
            this.decoder = (Decoder) ois.readObject();

            // Charger l'état de l'optimiseur
            int currentStep = (int) ois.readObject();
            int epoch = (int) ois.readObject();
            float learningRate = (float) ois.readObject();
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

    /**
     * Méthode de sérialisation personnalisée.
     *
     * @param oos Stream de sortie.
     * @throws IOException En cas d'erreur d'E/S.
     */
    private void writeObject(ObjectOutputStream oos) throws IOException {
        // Sauvegarder le chemin du fichier Word2Vec si nécessaire
        oos.writeObject(W2VECPATH);
    }

    /**
     * Méthode de désérialisation personnalisée.
     *
     * @param ois Stream d'entrée.
     * @throws IOException            En cas d'erreur d'E/S.
     * @throws ClassNotFoundException En cas de classe non trouvée lors de la
     *                                désérialisation.
     */
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        // Lire le chemin du fichier Word2Vec
        String word2vecPath = (String) ois.readObject();
        // Réinitialiser wordVectors
        this.wordVectors = WordVectorSerializer.loadStaticModel(new File(word2vecPath));
    }

    /**
     * Définit l'état d'entraînement du modèle.
     *
     * @param isTrained true si entraîné, false sinon.
     */
    public void setTrained(boolean isTrained) {
        this.isTrained = isTrained;
    }

    /**
     * Obtient la dimension du modèle.
     *
     * @return Dimension du modèle (dModel).
     */
    public int getDModel() {
        return dModel;
    }

    /**
     * Obtient la taille du vocabulaire.
     *
     * @return Taille du vocabulaire.
     */
    public int getVocabSize() {
        if (pretrainedEmbeddings == null) {
            throw new IllegalStateException("Les embeddings pré-entraînés ne sont pas initialisés.");
        }
        return (int) pretrainedEmbeddings.rows();
    }

    /**
     * Initialise les embeddings pré-entraînés.
     *
     * @param vocabSize Taille du vocabulaire.
     * @param dModel    Dimension du modèle.
     */
    public void initializeEmbeddings(int vocabSize, int dModel) {
        // Exemple : initialisation aléatoire des embeddings avec normalisation
        this.pretrainedEmbeddings = Nd4j.randn(DataType.FLOAT, vocabSize, dModel).divi(Math.sqrt(dModel));
    }

    /**
     * Récupère les embeddings pré-entraînés.
     *
     * @return INDArray contenant les embeddings [vocabSize, dModel]
     */
    public INDArray getPretrainedEmbeddings() {
        if (pretrainedEmbeddings == null) {
            throw new IllegalStateException(
                    "Les embeddings pré-entraînés ne sont pas initialisés. Appelez initializeEmbeddings() d'abord.");
        }
        return pretrainedEmbeddings;
    }

    public List<INDArray> getCombinedParameters() {
        return combinedParameters;
    }

    public List<INDArray> getCombinedGradients() {
        return combinedGradients;
    }

    // Méthode pour récupérer tous les paramètres du modèle
    public List<INDArray> getParameters() {
        return getCombinedParameters();
    }

    // Méthode pour récupérer les gradients
    public List<INDArray> getGradients() {
        return getCombinedGradients();
    }

    /**
     * Effectue une inférence sur un prompt donné.
     *
     * @param prompt    Texte d'entrée.
     * @param maxLength Longueur maximale de la séquence générée.
     * @return Texte généré.
     */
    public String infer(String prompt, int maxLength) {
        if (!isTrained) {
            throw new IllegalStateException("Le modèle doit être entraîné avant l'inférence.");
        }
    
        // Tokenisation du prompt
        List<String> promptTokens = tokenizer.tokenize(prompt);
        List<Integer> promptTokenIds = tokenizer.tokensToIds(promptTokens);
    
        // Ajouter le token de début si nécessaire
        promptTokenIds.add(0, tokenizer.getStartTokenId());
    
        // Convertir la liste en INDArray [1, seqLength]
        INDArray data = Nd4j.create(DataType.INT, 1, promptTokenIds.size());
        for (int j = 0; j < promptTokenIds.size(); j++) {
            data.putScalar(new int[] { 0, j }, promptTokenIds.get(j));
        }
    
        // Encoder le prompt (traité comme un batch de taille 1)
        Batch encoderBatch = new Batch(data, null, tokenizer);
        INDArray encodedPrompt = encoder.encode(false, encoderBatch); // [1,10,50]
    
        // Stocker les tokens d'entrée de l'encodeur pour la cross-attention
        INDArray encoderInputTokens = encoderBatch.getData(); // [1,10]
    
        // Initialiser les IDs de sortie avec le token de début
        List<Integer> outputIds = new ArrayList<>();
        outputIds.add(tokenizer.getStartTokenId());
    
        for (int i = 0; i < maxLength; i++) {
            // Convertir les IDs de sortie en INDArray [1, currentOutputLength]
            INDArray decoderInputIds = Nd4j.create(DataType.INT, 1, outputIds.size());
            for (int j = 0; j < outputIds.size(); j++) {
                decoderInputIds.putScalar(new int[] { 0, j }, outputIds.get(j));
            }
    
            // Créer un Batch pour le décodeur avec les données actuelles
            Batch decoderBatch = new Batch(decoderInputIds, null, tokenizer);
    
            // Encoder les IDs de sortie
            INDArray encodedDecoderInput = tokenizer.lookupEmbeddings(decoderInputIds); // [1, currentOutputLength, dModel]
    
            // Décoder en passant les tokens d'entrée de l'encodeur
            INDArray logits = decoder.decode(false, encodedPrompt, encodedDecoderInput, decoderBatch, encoderInputTokens); // [1,1, vocabSize]
    
            // Extraction des logits du dernier token généré
            int lastPosition = (int) logits.shape()[1] - 1; // seqLength - 1
            INDArray lastTokenLogits = logits.get(
                    NDArrayIndex.point(0), // batch 0
                    NDArrayIndex.point(lastPosition), // dernière position dans seqLength
                    NDArrayIndex.all() // tous les éléments dans vocabSize
            ).dup(); // [vocabSize]
    
            // Appliquer softmax pour obtenir les probabilités
            INDArray softmaxLogits = Transforms.softmax(lastTokenLogits, false); // [vocabSize]
    
            // Sélectionner le token avec la plus haute probabilité
            int predictedTokenId = Nd4j.argMax(softmaxLogits, 0).getInt(0);
    
            // Ajouter le token prédit à la séquence de sortie
            outputIds.add(predictedTokenId);
    
            // Vérification du token de fin
            if (predictedTokenId == tokenizer.getEndTokenId()) {
                break;
            }
        }
    
        // Conversion des IDs en tokens
        // Exclure le token de début
        List<Integer> generatedTokenIds = outputIds.subList(1, outputIds.size());
        return tokenizer.idsToTokens(generatedTokenIds);
    }

    /**
     * Pad une seule séquence à une longueur maximale avec le token de padding.
     *
     * @param sequence   La séquence d'IDs de tokens.
     * @param maxLength  La longueur maximale à atteindre.
     * @param padTokenId L'ID du token de padding.
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
     * Méthode pour afficher les relations d'attention pour chaque échantillon du
     * batch.
     *
     * @param batchInputQueryTokens Liste des listes de tokens de requête pour
     *                              chaque échantillon du batch.
     * @param batchInputKeyTokens   Liste des listes de tokens de clé pour chaque
     *                              échantillon du batch.
     */
    public void displayAttentionRelations(
            List<List<String>> batchInputQueryTokens,
            List<List<String>> batchInputKeyTokens,
            List<List<String>> batchTargetQueryTokens) {

        Map<Integer, String> idToTokenMap = tokenizer.getIdToTokenMap();

        // Vérifier que le nombre d'échantillons dans les queries et les keys correspond
        if (batchInputQueryTokens.size() != batchInputKeyTokens.size() ||
                batchInputQueryTokens.size() != batchTargetQueryTokens.size()) {
            throw new IllegalArgumentException(
                    "Le nombre d'échantillons dans les queries, les keys et les targets doit être identique.");
        }

        // Itérer sur chaque échantillon du batch
        for (int sampleIdx = 0; sampleIdx < batchInputQueryTokens.size(); sampleIdx++) {

            List<String> inputQueryTokens = batchInputQueryTokens.get(sampleIdx);
            List<String> inputKeyTokens = batchInputKeyTokens.get(sampleIdx);
            List<String> targetQueryTokens = batchTargetQueryTokens.get(sampleIdx);

            System.out.println("===== Échantillon " + (sampleIdx + 1) + " =====");

            // Afficher les poids d'attention de l'encodeur (self-attention)
            for (int i = 0; i < encoder.layers.size(); i++) {
                EncoderLayer layer = encoder.layers.get(i);
                MultiHeadAttention selfAttn = layer.getSelfAttention();
                System.out.println("===== Encoder Layer " + (i + 1) + " Self-Attention Weights =====");
                selfAttn.printAttentionWeights(inputQueryTokens, inputKeyTokens, sampleIdx, idToTokenMap);
            }

            // Afficher les poids d'attention du décodeur (self-attention et
            // cross-attention)
            for (int i = 0; i < decoder.layers.size(); i++) {
                DecoderLayer layer = decoder.layers.get(i);
                MultiHeadAttention selfAttn = layer.getSelfAttention();
                MultiHeadAttention crossAttn = layer.getCrossAttention();

                System.out.println("===== Decoder Layer " + (i + 1) + " Self-Attention Weights =====");
                selfAttn.printAttentionWeights(targetQueryTokens, targetQueryTokens, sampleIdx, idToTokenMap);

                System.out.println("===== Decoder Layer " + (i + 1) + " Cross-Attention Weights =====");
                // Pour le cross-attention, les queries proviennent du décodeur (target) et les
                // keys de l'encodeur (input)
                crossAttn.printAttentionWeights(targetQueryTokens, inputKeyTokens, sampleIdx, idToTokenMap);
            }

            System.out.println(); // Ligne vide entre les échantillons
        }
    }

    /**
     * Vérifie si le modèle a été entraîné.
     *
     * @return true si entraîné, false sinon.
     */
    public boolean isTrained() {
        return isTrained;
    }

    public void setOptimizer(CustomAdamOptimizer optimizer) {
        this.optimizer = optimizer;
    }

}
