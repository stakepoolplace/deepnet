package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Classe représentant le décodeur du modèle Transformer.
 */
public class Decoder implements Serializable {
    private static final long serialVersionUID = 283129978055526337L;
    List<DecoderLayer> layers;
    private LayerNorm layerNorm;
    private int numLayers;
    protected int dModel;
    private int numHeads;
    private double dropoutRate;
    private LinearProjection linearProjection; // Projection linéaire vers la taille du vocabulaire

    // Cache pour stocker les entrées des couches pendant la passe forward
    private List<INDArray> forwardCache;
    private INDArray lastNormalizedInput; // Stocke l'entrée normalisée après LayerNorm

    private Tokenizer tokenizer; // Référence au Tokenizer

    public Decoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, Tokenizer tokenizer) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dropoutRate = dropoutRate;
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.tokenizer = tokenizer;
        this.linearProjection = new LinearProjection(dModel, tokenizer.getVocabSize()); // Utiliser la taille du vocabulaire du Tokenizer
        this.forwardCache = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new DecoderLayer(dModel, numHeads, dff, dropoutRate));
            this.forwardCache.add(null); // Initialiser le cache avec des valeurs nulles
        }
    }

    /**
     * Réinitialise le cache. Appelé avant une nouvelle passe forward.
     */
    public void resetCache() {
        for (int i = 0; i < forwardCache.size(); i++) {
            forwardCache.set(i, null);
        }
    }

    /**
     * Décode les entrées en utilisant les couches du décodeur.
     *
     * @param isTraining       Indique si le modèle est en mode entraînement.
     * @param encoderOutput    Sortie de l'encodeur.
     * @param encodedDecoderInput Entrée encodée pour le décodeur.
     * @param lookAheadMask    Masque look-ahead pour l'auto-attention.
     * @param paddingMask      Masque de padding pour l'attention croisée.
     * @return Logits après projection linéaire.
     */
    public INDArray decode(boolean isTraining, INDArray encoderOutput, INDArray encodedDecoderInput, INDArray lookAheadMask, INDArray paddingMask) {
        // Réinitialiser le cache avant une nouvelle passe forward
        resetCache();

        // Passer à travers les couches du décodeur
        for (int i = 0; i < layers.size(); i++) {
            DecoderLayer layer = layers.get(i);
            encodedDecoderInput = layer.forward(isTraining, encodedDecoderInput, encoderOutput, lookAheadMask, paddingMask, forwardCache.get(i));
            forwardCache.set(i, encodedDecoderInput.dup()); // Stocker l'entrée actuelle dans le cache
        }

        // Normalisation finale
        encodedDecoderInput = layerNorm.forward(encodedDecoderInput);
        lastNormalizedInput = encodedDecoderInput.dup(); // Stocker l'entrée normalisée

        // Projection linéaire vers la taille du vocabulaire
        INDArray logits = linearProjection.project(encodedDecoderInput); // [batchSize, targetSeqLength, vocabSize]
        // System.out.println("Logits shape: " + Arrays.toString(logits.shape()));

        return logits;
    }
    
    /**
     * Passe backward à travers le décodeur.
     *
     * @param gradOutput Gradient provenant de la couche suivante (logits).
     * @return Gradients à propager vers l'encodeur.
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        // S'assurer qu'une passe forward a été effectuée
        if (lastNormalizedInput == null) {
            throw new IllegalStateException("L'entrée normalisée n'est pas initialisée. Effectuez une passe forward d'abord.");
        }

        // Backpropager à travers la projection linéaire
        Map<String, INDArray> gradLinearProjection = linearProjection.backward(lastNormalizedInput, gradOutput);

        // Backpropager à travers la normalisation de couche finale
        Map<String, INDArray> gradLayerNorm = layerNorm.backward(gradLinearProjection.get("input"));

        // Transformer gradLayerNorm en un Map pour correspondre à la signature attendue par DecoderLayer.backward
        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("input", gradLayerNorm.get("input")); // Utiliser "input" comme clé

        // Commencer avec le gradient à la sortie du Décodeur
        for (int i = layers.size() - 1; i >= 0; i--) {
            DecoderLayer layer = layers.get(i);
            // Passer le cache correspondant à cette couche
            INDArray layerInput = forwardCache.get(i);
            gradMap = layer.backward(gradMap, layerInput);
        }
        // À ce stade, gradMap contiendrait le gradient à propager à l'Encodeur

        return gradMap;
    }

    /**
     * Obtient tous les paramètres du décodeur.
     *
     * @return Liste des paramètres.
     */
    public List<INDArray> getParameters() {
        List<INDArray> params = new ArrayList<>();

        // Collecter les paramètres de chaque couche du décodeur
        for (DecoderLayer layer : layers) {
            params.addAll(layer.getParameters());
        }

        // Collecter les paramètres de la normalisation de couche finale
        params.addAll(layerNorm.getParameters());

        // Collecter les paramètres de la projection linéaire
        params.addAll(linearProjection.getParameters());

        return params;
    }

    /**
     * Obtient tous les gradients du décodeur.
     *
     * @return Liste des gradients.
     */
    public List<INDArray> getGradients() {
        List<INDArray> grads = new ArrayList<>();

        // Collecter les gradients de chaque couche du décodeur
        for (DecoderLayer layer : layers) {
            grads.addAll(layer.getGradients());
        }

        // Collecter les gradients de la normalisation de couche finale
        grads.addAll(layerNorm.getGradients());

        // Collecter les gradients de la projection linéaire
        grads.addAll(linearProjection.getGradients());

        return grads;
    }

    /**
     * Obtient le nombre total de paramètres dans le décodeur.
     *
     * @return Nombre de paramètres.
     */
    public int getNumberOfParameters() {
        int numParams = 0;

        // Parcourir toutes les couches de décodeur pour compter leurs paramètres
        for (DecoderLayer layer : layers) {
            numParams += layer.getNumberOfParameters();
        }

        // Ajouter les paramètres de la normalisation de couche et de la projection linéaire
        numParams += layerNorm.getNumberOfParameters();
        numParams += linearProjection.getNumberOfParameters();

        return numParams;
    }

    /**
     * Classe interne représentant une couche unique du décodeur.
     */
    static class DecoderLayer implements Serializable {
        private static final long serialVersionUID = 4450374170745550258L;
        MultiHeadAttention selfAttention;
        MultiHeadAttention encoderDecoderAttention;
        PositionwiseFeedForward feedForward;
        LayerNorm layerNorm1;
        LayerNorm layerNorm2;
        LayerNorm layerNorm3;
        Dropout dropout1;
        Dropout dropout2;
        Dropout dropout3;

        public DecoderLayer(int dModel, int numHeads, int dff, double dropoutRate) {
            this.selfAttention = new MultiHeadAttention(dModel, numHeads);
            this.encoderDecoderAttention = new MultiHeadAttention(dModel, numHeads);
            this.feedForward = new PositionwiseFeedForward(dModel, dff);
            this.layerNorm1 = new LayerNorm(dModel);
            this.layerNorm2 = new LayerNorm(dModel);
            this.layerNorm3 = new LayerNorm(dModel);
            this.dropout1 = new Dropout(dropoutRate);
            this.dropout2 = new Dropout(dropoutRate);
            this.dropout3 = new Dropout(dropoutRate);
        }

        public MultiHeadAttention getSelfAttention() {
            return this.selfAttention;
        }
        
        public MultiHeadAttention getCrossAttention() {
            return this.encoderDecoderAttention;
        }

        /**
         * Passe forward à travers la couche du décodeur.
         *
         * @param isTraining       Indique si le modèle est en mode entraînement.
         * @param x                Entrée actuelle.
         * @param encoderOutput    Sortie de l'encodeur pour l'attention croisée.
         * @param lookAheadMask    Masque look-ahead pour l'auto-attention.
         * @param paddingMask      Masque de padding pour l'attention croisée.
         * @param cachedInput      Entrée mise en cache de la passe forward précédente.
         * @return Sortie de la couche.
         */
        public INDArray forward(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask, INDArray cachedInput) {
            // Self-attention
            INDArray attn1 = selfAttention.forward(x, x, x, lookAheadMask);
            attn1 = dropout1.apply(isTraining, attn1);
            x = layerNorm1.forward(x.add(attn1)); // Add & Norm

            // Encoder-decoder attention
            INDArray attn2 = encoderDecoderAttention.forward(x, encoderOutput, encoderOutput, paddingMask);
            attn2 = dropout2.apply(isTraining, attn2);
            x = layerNorm2.forward(x.add(attn2)); // Add & Norm

            // Feed-forward
            INDArray ffOutput = feedForward.forward(x);
            ffOutput = dropout3.apply(isTraining, ffOutput);
            return layerNorm3.forward(x.add(ffOutput)); // Add & Norm again
        }

        /**
         * Passe backward à travers la couche du décodeur.
         *
         * @param gradOutput Gradient provenant de la couche suivante.
         * @param cachedInput Entrée mise en cache de la passe forward précédente.
         * @return Gradient à propager vers la couche précédente.
         */
        public Map<String, INDArray> backward(Map<String, INDArray> gradOutput, INDArray cachedInput) {
            Map<String, INDArray> gradients = new HashMap<>();

            // Backpropager à travers LayerNorm3
            Map<String, INDArray> gradLayerNorm3 = layerNorm3.backward(gradOutput.get("input"));
            gradients.putAll(gradLayerNorm3);

            // Backpropager à travers FeedForward
            Map<String, INDArray> gradFeedForward = feedForward.backward(gradLayerNorm3.get("input"));
            gradients.putAll(gradFeedForward);

            // Backpropager à travers LayerNorm2
            Map<String, INDArray> gradLayerNorm2 = layerNorm2.backward(gradFeedForward.get("input"));
            gradients.putAll(gradLayerNorm2);

            // Backpropager à travers EncoderDecoderAttention
            Map<String, INDArray> gradEncoderDecoderAttention = encoderDecoderAttention.backward(gradLayerNorm2.get("input"));
            gradients.putAll(gradEncoderDecoderAttention);

            // Backpropager à travers LayerNorm1
            Map<String, INDArray> gradLayerNorm1 = layerNorm1.backward(gradEncoderDecoderAttention.get("input"));
            gradients.putAll(gradLayerNorm1);

            // Backpropager à travers SelfAttention
            Map<String, INDArray> gradSelfAttention = selfAttention.backward(gradLayerNorm1.get("input"));
            gradients.putAll(gradSelfAttention);

            return gradients;
        }

        /**
         * Obtient tous les paramètres de la couche du décodeur.
         *
         * @return Liste des paramètres.
         */
        public List<INDArray> getParameters() {
            List<INDArray> layerParams = new ArrayList<>();

            layerParams.addAll(selfAttention.getParameters());
            layerParams.addAll(encoderDecoderAttention.getParameters());
            layerParams.addAll(feedForward.getParameters());
            layerParams.addAll(layerNorm1.getParameters());
            layerParams.addAll(layerNorm2.getParameters());
            layerParams.addAll(layerNorm3.getParameters());

            return layerParams;
        }

        /**
         * Obtient tous les gradients de la couche du décodeur.
         *
         * @return Liste des gradients.
         */
        public List<INDArray> getGradients() {
            List<INDArray> layerGrads = new ArrayList<>();

            layerGrads.addAll(selfAttention.getGradients());
            layerGrads.addAll(encoderDecoderAttention.getGradients());
            layerGrads.addAll(feedForward.getGradients());
            layerGrads.addAll(layerNorm1.getGradients());
            layerGrads.addAll(layerNorm2.getGradients());
            layerGrads.addAll(layerNorm3.getGradients());

            return layerGrads;
        }

        /**
         * Obtient le nombre total de paramètres dans la couche du décodeur.
         *
         * @return Nombre de paramètres.
         */
        public long getNumberOfParameters() {
            return selfAttention.getNumberOfParameters() +
                   encoderDecoderAttention.getNumberOfParameters() +
                   feedForward.getNumberOfParameters() +
                   layerNorm1.getNumberOfParameters() +
                   layerNorm2.getNumberOfParameters() +
                   layerNorm3.getNumberOfParameters();
        }

        /**
         * Obtient le nombre total de gradients dans la couche du décodeur.
         *
         * @return Nombre de gradients.
         */
        public long getNumberOfGradients() {
            return selfAttention.getNumberOfGradients() +
                   encoderDecoderAttention.getNumberOfGradients() +
                   feedForward.getNumberOfGradients() +
                   layerNorm1.getNumberOfGradients() +
                   layerNorm2.getNumberOfGradients() +
                   layerNorm3.getNumberOfGradients();
        }
    }
}
