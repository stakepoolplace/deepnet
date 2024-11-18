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
        List<String> defaultVocab = Arrays.asList("hello", "world", "test", "input", "output", "<PAD>", "<UNK>", "<START>", "<END>");
        this.tokenizer = new Tokenizer(defaultVocab, dModel, maxSequenceLength);
        this.pretrainedEmbeddings = this.tokenizer.getPretrainedEmbeddings();
        
        // Initialiser les autres composants
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);

        addCombinedParameters();

        this.optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, combinedParameters); // Initialisation hypothétique
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
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, float initialLr, int warmupSteps) {
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
            // Si les WordVectors ne sont pas chargés, initialiser avec un vocabulaire vide ou par défaut
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
    }

    /**
     * Nouveau constructeur compatible avec le test.
     *
     * @param numLayers    Nombre de couches dans l'encodeur et le décodeur.
     * @param dModel       Dimension du modèle.
     * @param numHeads     Nombre de têtes dans l'attention multi-têtes.
     * @param dff          Dimension de la couche Feed-Forward.
     * @param dropoutRate  Taux de dropout.
     * @param vocabSize    Taille du vocabulaire.
     * @param tokenizer    Instance de Tokenizer personnalisée.
     */
    public TransformerModel(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, int vocabSize, Tokenizer tokenizer, float initialLr, int warmupSteps) {
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
            tokenizer.setPretrainedEmbeddings(this.pretrainedEmbeddings); // Assurez-vous que Tokenizer peut définir les embeddings
        }

        // Initialiser l'encodeur et le décodeur avec le tokenizer personnalisé
        this.encoder = new Encoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dff, dropoutRate, this.tokenizer);

        addCombinedParameters();

        // Initialiser l'optimiseur avec les paramètres combinés
        this.optimizer = new CustomAdamOptimizer(initialLr, dModel, warmupSteps, combinedParameters);
    }

    // Méthode pour initialiser l'optimiseur après avoir collecté tous les paramètres
    public void initializeOptimizer(float learningRate, int warmupSteps) {
        List<INDArray> combinedParameters = this.encoder.getParameters();
        combinedParameters.addAll(this.decoder.getParameters());
        this.optimizer = new CustomAdamOptimizer(learningRate, dModel, warmupSteps, combinedParameters);
    }

    /**
     * Entraîne le modèle sur un nombre donné d'epochs.
     *
     * @param dataGenerator Générateur de données pour l'entraînement.
     * @param epochNum      Nombre d'epochs à entraîner.
     * @throws IOException En cas d'erreur d'E/S.
     */
    public void train(DataGenerator dataGenerator, int epochNum) throws IOException {
        
        freezeSpecialTokenEmbeddings();

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
                INDArray data = batch.getData();     // [batchSize, seqLength]
                INDArray target = batch.getTarget(); // [batchSize, seqLength]
                // INDArray mask = batch.getMask();     // [batchSize, 1, 1, seqLength] (Non utilisé directement ici)

                // Créer les masques de padding pour l'encodeur et le décodeur
                INDArray encoderPaddingMask = createPaddingMask(data);       // [batchSize, 1, 1, seqLength]
                //INDArray decoderPaddingMask = createPaddingMask(target);    // [batchSize, 1, 1, seqLength]

                // Créer le masque look-ahead pour le décodeur (taille basée sur la séquence cible)
                // Créer le masque look-ahead avec le batch size
                int batchSize = (int) data.shape()[0];
                int targetSeqLength = (int) target.shape()[1];
                INDArray lookAheadMask = createLookAheadMask(batchSize, targetSeqLength); // [1, 1, seqLength, seqLength]

                // Encoder les données du batch
                INDArray encoded = encoder.encode(true, data, encoderPaddingMask); // [batchSize, seqLength, dModel]

                // Décoder les données encodées
                INDArray decodedOutput = decoder.decode(true, encoded, encoded, lookAheadMask, encoderPaddingMask); // [batchSize, seqLength, dModel]

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
                Map<String, INDArray> decoderGradients = decoder.backward(initialGradients);
                Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);
                encoder.backward(encoderGradients);

                // Ajouter tous les gradients calculés aux `combinedGradients`
                addCombinedGradients();

                // Clip gradients
                clipGradients(combinedGradients, 0.5f); // Exemple avec maxNorm = 0.5

                // Mettre à jour les poids du modèle via l'optimiseur
                optimizer.update(combinedParameters, combinedGradients);

                // for (int i = 0; i < combinedGradients.size(); i++) {
                //     INDArray grad = combinedGradients.get(i);
                //     if (grad == null) {
                //         throw new IllegalStateException("Le gradient " + i + " est nul.");
                //     }
                // }
                

                // for (int i = 0; i < combinedParameters.size(); i++) {
                //     INDArray paramBefore = combinedParameters.get(i).dup();
                //     optimizer.update(combinedParameters, combinedGradients);
                //     INDArray paramAfter = combinedParameters.get(i);
                //     System.out.println("Param " + i + " avant: " + paramBefore);
                //     System.out.println("Param " + i + " après: " + paramAfter);
                //     //assertFalse(paramBefore.equalsWithEps(paramAfter, 1e-6), "Le paramètre " + i + " devrait avoir été mis à jour.");
                // }



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

   /**
     * Méthode d'entraînement pour un epoch, retourne la perte moyenne.
     *
     * @param dataGenerator Générateur de données pour l'entraînement.
     * @return Perte moyenne sur l'epoch.
     * @throws IOException En cas d'erreur d'E/S.
     */
    public float trainEpochAndGetLoss(DataGenerator dataGenerator) throws IOException {
        return trainEpoch(dataGenerator);
    }

    // public float trainEpoch(DataGenerator dataGenerator) throws IOException {
    //     float totalLoss = 0.0f;
    //     int totalTokens = 0;
    
    //     // Initialiser pour l'epoch
    //     optimizer.setEpoch(optimizer.getEpoch() + 1);
    //     System.out.println("Epoch " + optimizer.getEpoch());
    
    //     while (dataGenerator.hasNextBatch()) {
    //         // Nettoyer les gradients précédents
    //         cleanGradients();
    
    //         Batch batch = dataGenerator.nextBatch();
    //         INDArray data = batch.getData();
    //         INDArray target = batch.getTarget();
    
    //         // Créer les masques de padding pour le batch
    //         INDArray encoderPaddingMask = createPaddingMask(data);
    //         INDArray decoderPaddingMask = createPaddingMask(target);
    //         INDArray lookAheadMask = createLookAheadMask((int) data.shape()[0], (int) target.shape()[1]);
    
    //         // Encoder les données du batch
    //         INDArray encoded = encoder.encode(true, data, encoderPaddingMask);
    
    //         // Décoder les données encodées
    //         INDArray decodedOutput = decoder.decode(true, encoded, encoded, lookAheadMask, encoderPaddingMask);
    
    //         // Calculer la perte et les gradients
    //         List<INDArray> decodedLogits = Arrays.asList(decodedOutput);
    //         Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, target);
    //         float loss = lossAndGradients.getLeft();
    //         INDArray initialGradients = lossAndGradients.getRight();
    
    //         totalLoss += loss;
    //         totalTokens += target.sumNumber().intValue();
    
    //         // Afficher la perte pour le monitoring
    //         System.out.println("Perte pour ce batch: " + loss);
    
    //         // Étape 2: Rétropropagation à travers le Décodeur
    //         Map<String, INDArray> decoderGradients = decoder.backward(initialGradients);
    
    //         // Extraire les gradients pertinents pour l'encodeur à partir de decoderGradients
    //         Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);
    
    //         // Étape 3: Rétropropagation à travers l'Encodeur
    //         encoder.backward(encoderGradients);
    
    //         // Ajouter tous les gradients calculés aux `combinedGradients`
    //         addCombinedGradients();
    //     }
    
    //     // Calculer la perte moyenne
    //     float averageLoss = totalLoss / totalTokens;
    
    //     // Mettre à jour les poids du modèle via l'optimiseur
    //     optimizer.update(combinedParameters, combinedGradients);
    
    //     // Réinitialiser le générateur de données pour le prochain epoch
    //     dataGenerator.reset();
    //     System.out.println("Epoch " + optimizer.getEpoch() + " completed with average loss: " + averageLoss);
    
    //     // Marquer le modèle comme entraîné après le premier epoch
    //     if (optimizer.getEpoch() >= 1) {
    //         isTrained = true;
    //     }
    
    //     return averageLoss;
    // }

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
            INDArray data = batch.getData();   // Utiliser directement le format INDArray
            INDArray target = batch.getTarget();  // Utiliser directement le format INDArray
            // INDArray mask = batch.getMask();  // Masque de padding pour les séquences
    
            // Créer les masques de padding pour le batch
            INDArray encoderPaddingMask = createPaddingMask(data);
            //INDArray decoderPaddingMask = createPaddingMask(target);
            // Créer le masque look-ahead avec le batch size
            int batchSize = (int) data.shape()[0];
            int targetSeqLength = (int) target.shape()[1];
            INDArray lookAheadMask = createLookAheadMask(batchSize, targetSeqLength); // Longueur max du target
    
            // System.out.println("Encoder Mask: " + encoderPaddingMask);
            // System.out.println("Decoder Mask: " + decoderPaddingMask);
            // System.out.println("Look-Ahead Mask: " + lookAheadMask);

            // Encoder les données du batch
            INDArray encoded = encoder.encode(true, data, encoderPaddingMask);
            // System.out.println("Encoded output shape: " + Arrays.toString(encoded.shape()));
    
            // Décoder les données encodées
            INDArray decodedOutput = decoder.decode(true, encoded, encoded, lookAheadMask, encoderPaddingMask);
            // System.out.println("Decoded output shape: " + Arrays.toString(decodedOutput.shape()));
    
            // Calculer la perte et les gradients
            List<INDArray> decodedLogits = Arrays.asList(decodedOutput);
            Pair<Float, INDArray> lossAndGradients = calculateCrossEntropyLossAndGradient(decodedLogits, target);
            float loss = lossAndGradients.getLeft();
            INDArray initialGradients = lossAndGradients.getRight();
    
            totalLoss += loss;
            totalTokens += target.sumNumber().intValue(); // Total de tokens cibles pour normaliser la perte
    
            // Afficher la perte pour le monitoring
            System.out.println("Perte pour ce batch: " + loss);
    
            // Étape 2: Rétropropagation à travers le Décodeur
            Map<String, INDArray> decoderGradients = decoder.backward(initialGradients);
    
            // Extraire les gradients pertinents pour l'encodeur à partir de decoderGradients
            Map<String, INDArray> encoderGradients = extractEncoderGradients(decoderGradients);
    
            // Étape 3: Rétropropagation à travers l'Encodeur
            encoder.backward(encoderGradients);
    
            // Ajouter tous les gradients calculés aux `combinedGradients`
            addCombinedGradients();

            for (INDArray grad : combinedGradients) {
                if (grad.isNaN().any() || grad.isInfinite().any()) {
                    throw new RuntimeException("Gradient contains NaN or Infinite values.");
                }
            }


            // for (int i = 0; i < combinedGradients.size(); i++) {
            //     INDArray grad = combinedGradients.get(i);
            //     double norm = grad.norm2Number().doubleValue();
            //     System.out.println("Gradient " + i + " Norm: " + norm);
            //     if (norm < 1e-6) {
            //         throw new RuntimeException("Gradient " + i + " est trop petit.");
            //     }
            // }

            // for (int i = 0; i < combinedParameters.size(); i++) {
            //         INDArray paramBefore = combinedParameters.get(i).dup();
            //         // Mise à jour des paramètres
            //         optimizer.update(combinedParameters, combinedGradients);
            //         INDArray paramAfter = combinedParameters.get(i);
            //         System.out.println("Param " + i + " Before Update: " + paramBefore);
            //         System.out.println("Param " + i + " After Update: " + paramAfter);
                    
            //     }

            // Avant l'appel à l'optimiseur
            clipGradients(combinedGradients, 0.5);

            // Mettre à jour les poids du modèle via l'optimiseur
            optimizer.update(combinedParameters, combinedGradients);
    
            // (Optionnel) Afficher la progression
            // System.out.println("Batch processed.");
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

    // private void clipGradients(List<INDArray> gradients, double maxNorm) {
    //     double totalNorm = 0.0;
    //     for (INDArray grad : gradients) {
    //         totalNorm += Math.pow(grad.norm2Number().doubleValue(), 2);
    //     }
    //     totalNorm = Math.sqrt(totalNorm);
    //     if (totalNorm > maxNorm) {
    //         double scale = maxNorm / totalNorm;
    //         for (INDArray grad : gradients) {
    //             grad.muli(scale);
    //         }
    //         // System.out.println("Gradients clipped. Scale factor: " + scale);
    //     }
    // }

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
     * @param sequences   Liste de séquences d'IDs.
     * @param maxLength   Longueur maximale à atteindre.
     * @param padTokenId  ID du token de padding.
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
     * Crée un masque de padding pour un batch donné.
     *
     * @param data INDArray contenant les IDs de tokens du batch [batchSize, seqLength].
     * @return INDArray représentant le masque de padding [batchSize, 1, 1, seqLength].
     */
    public INDArray createPaddingMask(INDArray tokens) {
        // tokens : [batchSize, seqLength]
        // output mask : [batchSize, 1, 1, seqLength]
        
        // Identifier les positions de padding
        INDArray paddingPositions = tokens.eq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT); // [batchSize, seqLength]
        System.out.println("Padding Positions:\n" + paddingPositions);
        
        // Créer un masque initial rempli de 0.0f
        INDArray paddingMask = Nd4j.zeros(DataType.FLOAT, tokens.size(0), 1, 1, tokens.size(1)); // [batchSize, 1, 1, seqLength]
        
        // Boucler sur chaque élément pour remplacer 1.0f par -Infinity
        for (int i = 0; i < tokens.size(0); i++) { // Itérer sur batchSize
            for (int j = 0; j < tokens.size(1); j++) { // Itérer sur seqLength
                if (paddingPositions.getFloat(i, j) == 1.0f) {
                    paddingMask.putScalar(new int[]{i, 0, 0, j}, Float.NEGATIVE_INFINITY);
                } else {
                    paddingMask.putScalar(new int[]{i, 0, 0, j}, 0.0f);
                }
            }
        }
        
        System.out.println("Generated Padding Mask:\n" + paddingMask);
        return paddingMask;
    }


    /**
     * Crée un masque look-ahead pour le décodeur.
     *
     * @param size Taille de la séquence.
     * @return INDArray représentant le masque look-ahead [1, 1, size, size].
     */
    public INDArray createLookAheadMask(int batchSize, int size) {
        // Créer une matrice triangulaire inférieure remplie de 0.0f
        INDArray lookAheadMask = Nd4j.zeros(DataType.FLOAT, size, size); // [size, size]
        
        // Remplir le masque avec 0.0f où j <= i et -Infinity où j > i
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (j > i) {
                    lookAheadMask.putScalar(new int[]{i, j}, Float.NEGATIVE_INFINITY);
                } else {
                    lookAheadMask.putScalar(new int[]{i, j}, 0.0f);
                }
            }
        }
        
        // Reshaper pour correspondre aux dimensions attendues [1, 1, size, size]
        lookAheadMask = lookAheadMask.reshape(1, 1, size, size); // [1, 1, size, size]
        
        // Répéter le masque pour chaque exemple du batch
        INDArray repeatedMask = Nd4j.tile(lookAheadMask, batchSize, 1, 1, 1); // [batchSize, 1, size, size]
        
        return repeatedMask;
    }
    

    /**
     * Calcule la perte d'entropie croisée et les gradients associés.
     *
     * @param decodedLogits Logits générés par le décodeur [batchSize, seqLength, vocabSize].
     * @param targetBatch   INDArray contenant les IDs des tokens cibles [batchSize, seqLength].
     * @return Un Pair contenant la perte moyenne et les gradients [batchSize, seqLength, vocabSize].
     */
    protected Pair<Float, INDArray> calculateCrossEntropyLossAndGradient(List<INDArray> decodedLogits, INDArray targetBatch) {
        INDArray logits = decodedLogits.get(0); // [batchSize, seqLength, vocabSize]
        int batchSize = (int) logits.shape()[0];
        int seqLength = (int) logits.shape()[1];
        int vocabSize = (int) logits.shape()[2];
    
        INDArray probabilities = NDArrayUtils.stableSoftmax(logits, 2); // [batchSize, seqLength, vocabSize]
        
        // Créer une INDArray one-hot pour les cibles
        INDArray targetOneHot = Nd4j.zeros(DataType.FLOAT, batchSize, seqLength, vocabSize);
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLength; i++) {
                int targetId = targetBatch.getInt(b, i);
                targetOneHot.putScalar(new int[] { b, i, targetId }, 1.0f);
            }
        }
    
        // Créer un masque pour ignorer les `<PAD>`
        INDArray paddingMask = createPaddingMask(targetBatch); // [batchSize, 1, 1, seqLength]
        INDArray paddingMaskReshaped = paddingMask.reshape(batchSize, 1, seqLength); // [batchSize, 1, seqLength]
    
        // Calculer la perte d'entropie croisée en masquant les `<PAD>`
        INDArray logSoftmax = Transforms.log(probabilities.add(1e-10)); // éviter log(0)
        INDArray crossEntropy = logSoftmax.mul(targetOneHot).neg(); // [batchSize, seqLength, vocabSize]
        INDArray maskedCrossEntropy = crossEntropy.mul(paddingMaskReshaped.broadcast(batchSize, vocabSize, seqLength).permute(0, 2, 1));
        
        // Corriger la division en convertissant le masque booléen en FLOAT
        float loss = maskedCrossEntropy.sumNumber().floatValue() / targetBatch.neq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT).sumNumber().floatValue();

        // Calculer les gradients (softmax - targetOneHot) en masquant les `<PAD>`
        INDArray gradients = probabilities.sub(targetOneHot)
            .mul(paddingMaskReshaped.broadcast(batchSize, vocabSize, seqLength).permute(0, 2, 1))
            .div(targetBatch.neq(tokenizer.getPadTokenId()).castTo(DataType.FLOAT).sumNumber().floatValue());
    
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
            }
        }
    
        // Ajoute les paramètres du decoder
        List<INDArray> decoderParams = decoder.getParameters();
        for (INDArray param : decoderParams) {
            if (!isSpecialTokenParameter(param)) {
                combinedParameters.add(param);
            }
        }
    }
    
    private boolean isSpecialTokenParameter(INDArray param) {
        // Implémenter une logique pour vérifier si le param correspond à un token spécial
        // Par exemple, comparer les adresses des objets ou les indices des embeddings
        // Cette implémentation dépend de votre structure de paramètres
        // Voici un exemple simplifié :
        return param.equals(tokenizer.getPretrainedEmbeddings().getRow(tokenizer.getPadTokenId())) ||
               param.equals(tokenizer.getPretrainedEmbeddings().getRow(tokenizer.getStartTokenId())) ||
               param.equals(tokenizer.getPretrainedEmbeddings().getRow(tokenizer.getEndTokenId())) ||
               param.equals(tokenizer.getPretrainedEmbeddings().getRow(tokenizer.getUnkTokenId()));
    }

    public void freezeSpecialTokenEmbeddings() {
        int[] specialTokenIds = { tokenizer.getPadTokenId(), tokenizer.getStartTokenId(), tokenizer.getEndTokenId(), tokenizer.getUnkTokenId() };
        
        for (int tokenId : specialTokenIds) {
            INDArray embedding = tokenizer.getPretrainedEmbeddings().getRow(tokenId);
            embedding.assign(Nd4j.zeros(dModel)); // Exemple pour <PAD>
            // Ou assigner une valeur spécifique
        }
    }

    /**
     * Ajoute les gradients de l'encodeur et du décodeur à la liste combinée.
     */
    private void addCombinedGradients() {
        // Ajoute les gradients de l'encoder
        combinedGradients.addAll(encoder.getGradients());

        // Ajoute les gradients du decoder
        combinedGradients.addAll(decoder.getGradients());
    }

    /**
     * Nettoie les gradients accumulés.
     */
    public void cleanGradients() {
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
     * @throws ClassNotFoundException En cas de classe non trouvée lors de la désérialisation.
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
     * @throws ClassNotFoundException En cas de classe non trouvée lors de la désérialisation.
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

        // Déterminer la longueur maximale de séquence après ajout du token de début
        int maxSeqLength = promptTokenIds.size();

        // Padder la séquence si nécessaire
        List<Integer> paddedPromptTokenIds = padSequencesSingle(promptTokenIds, maxSeqLength, tokenizer.getPadTokenId());

        // Convertir la liste en INDArray [1, seqLength]
        INDArray data = Nd4j.create(DataType.INT, 1, maxSeqLength);
        for (int j = 0; j < paddedPromptTokenIds.size(); j++) {
            data.putScalar(new int[] {0, j}, (int) paddedPromptTokenIds.get(j));
        }        


        // Créer le masque de padding pour l'encodeur
        INDArray encoderPaddingMask = createPaddingMask(data); // [1, 1, 1, seqLength]

        // Encoder le prompt (traité comme un batch de taille 1)
        INDArray encodedPrompt = encoder.encode(false, data, encoderPaddingMask); // [1, seqLength, dModel]

        // Initialiser les IDs de sortie avec le token de début
        List<Integer> outputIds = new ArrayList<>();
        outputIds.add(tokenizer.getStartTokenId());

        for (int i = 0; i < maxLength; i++) {
            // Convertir les IDs de sortie en INDArray [1, currentOutputLength]
            INDArray decoderInput = Nd4j.create(DataType.INT, 1, outputIds.size());

            for (int j = 0; j < outputIds.size(); j++) {
                decoderInput.putScalar(new int[] {0, j}, outputIds.get(j));
            }

            // Créer les masques pour le décodeur
            INDArray lookAheadMask = createLookAheadMask(1, outputIds.size()); // [1, 1, size, size]
            INDArray crossAttnMask = encoderPaddingMask; // [1, 1, 1, seqLength]

            // Encoder les IDs de sortie 
            INDArray encodedDecoderInput = tokenizer.lookupEmbeddings(decoderInput); // [1, currentOutputLength, dModel]

            // Décoder
            INDArray logits = decoder.decode(false, encodedPrompt, encodedDecoderInput, lookAheadMask, crossAttnMask); // [1, currentOutputLength, vocabSize]

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
        return tokenizer.idsToTokens(outputIds.subList(1, outputIds.size()));
    }

    /**
     * Pad une seule séquence à une longueur maximale avec le token de padding.
     *
     * @param sequence    La séquence d'IDs de tokens.
     * @param maxLength   La longueur maximale à atteindre.
     * @param padTokenId  L'ID du token de padding.
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
     * Méthode pour afficher les relations entre les tokens en utilisant les poids
     * d'attention stockés.
     *
     * @param inputTokens Liste des tokens d'entrée.
     */
    public void displayAttentionRelations(List<String> inputTokens) {
        // Afficher les poids d'attention de l'encodeur
        for (int i = 0; i < encoder.layers.size(); i++) {
            EncoderLayer layer = encoder.layers.get(i);
            MultiHeadAttention selfAttn = layer.getSelfAttention();
            System.out.println("===== Encoder Layer " + (i + 1) + " Attention Weights =====");
            selfAttn.printAttentionWeights(inputTokens);
        }

        // Afficher les poids d'attention du décodeur
        for (int i = 0; i < decoder.layers.size(); i++) {
            DecoderLayer layer = decoder.layers.get(i);
            MultiHeadAttention selfAttn = layer.getSelfAttention();
            MultiHeadAttention crossAttn = layer.getCrossAttention();

            System.out.println("===== Decoder Layer " + (i + 1) + " Self Attention Weights =====");
            selfAttn.printAttentionWeights(inputTokens);

            System.out.println("===== Decoder Layer " + (i + 1) + " Cross Attention Weights =====");
            crossAttn.printAttentionWeights(inputTokens);
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
