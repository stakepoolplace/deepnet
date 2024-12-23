package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

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
    LinearProjection linearProjection;
    private PositionalEncoding positionalEncoding;

    // Cache pour stocker les entrées des couches pendant la passe forward
    private List<INDArray> forwardCache;
    private INDArray lastNormalizedInput;

    private Tokenizer tokenizer;

    private double attentionDropout = 0.0;

    private float positionalEncodingScale = 1.0f;

    public Decoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, Tokenizer tokenizer, boolean useLayerNorm) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dropoutRate = dropoutRate;
        this.tokenizer = tokenizer;
        this.layers = new ArrayList<>();
        this.layerNorm = useLayerNorm ? new LayerNorm(dModel) : null;
        this.linearProjection = new LinearProjection(dModel, tokenizer.getVocabSize(), useLayerNorm);
        this.positionalEncoding = new PositionalEncoding(dModel);
        this.forwardCache = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new DecoderLayer(this, dModel, numHeads, dff, dropoutRate, useLayerNorm));
            this.forwardCache.add(null);
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
     * Utilisé par train() mais aussi par infer()
     *
     * @param isTraining       Indique si le modèle est en mode entraînement.
     * @param encoderOutput    Sortie de l'encodeur.
     * @param encodedDecoderInput Entrée encodée pour le décodeur.
     * @param batch              Batch contenant les données d'entrée et de sortie.
     * @param encoderInputTokens Tokens encodés de l'encodeur.
     * @return Logits après projection linéaire.
     */
    public INDArray decode(boolean isTraining, 
                          INDArray encoderOutput, 
                          INDArray encodedDecoderInput, 
                          Batch batch,
                          INDArray lookAheadMask,
                          INDArray queryPaddingMaskFromSource,
                          INDArray keyPaddingMaskFromSource,
                          INDArray queryPaddingMaskFromTarget,
                          INDArray keyPaddingMaskFromTarget) {
        // Réinitialiser le cache avant une nouvelle passe forward
        resetCache();

        // Lookup embeddings et ajouter le positional encoding
        INDArray inputEmbeddings = tokenizer.lookupEmbeddings(batch.getData());
        INDArray posEncoding = positionalEncoding.getPositionalEncoding(batch.getData().shape()[1]);
        posEncoding = posEncoding.reshape(1, batch.getData().shape()[1], dModel).broadcast(inputEmbeddings.shape());
        encodedDecoderInput = inputEmbeddings.add(posEncoding);

        // Passer à travers les couches du décodeur
        for (int i = 0; i < layers.size(); i++) {
            DecoderLayer layer = layers.get(i);
            encodedDecoderInput = layer.forward(
                isTraining,
                encodedDecoderInput,
                encoderOutput,
                lookAheadMask,
                forwardCache.get(i),
                queryPaddingMaskFromSource,
                keyPaddingMaskFromSource,
                queryPaddingMaskFromTarget,
                keyPaddingMaskFromTarget
            );
            forwardCache.set(i, encodedDecoderInput.dup());
        }

        // Normalisation finale et projection linéaire
        encodedDecoderInput = layerNorm != null ? layerNorm.forward(encodedDecoderInput) : encodedDecoderInput;
        lastNormalizedInput = encodedDecoderInput.dup();
        
        return linearProjection.project(encodedDecoderInput);
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
        Map<String, INDArray> gradLayerNorm = layerNorm != null ? layerNorm.backward(gradLinearProjection.get("input")) : gradLinearProjection;

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
        if (layerNorm != null) {
            params.addAll(layerNorm.getParameters());
        }

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
        if (layerNorm != null) {
            grads.addAll(layerNorm.getGradients());
        }

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
        if (layerNorm != null) {
            numParams += layerNorm.getNumberOfParameters();
        }
        numParams += linearProjection.getNumberOfParameters();

        return numParams;
    }

    /**
     * Classe interne représentant une couche unique du décodeur.
     */
    public static class DecoderLayer implements Serializable {
        private static final long serialVersionUID = 4450374170745550258L;
        Decoder decoder;
        MultiHeadAttention selfAttention;
        MultiHeadAttention encoderDecoderAttention;
        PositionwiseFeedForward feedForward;
        LayerNorm layerNorm1;
        LayerNorm layerNorm2;
        LayerNorm layerNorm3;
        Dropout dropout1;
        Dropout dropout2;
        Dropout dropout3;

        public DecoderLayer(Decoder decoder, int dModel, int numHeads, int dff, double dropoutRate, boolean useLayerNorm) {
            this.decoder = decoder;
            this.selfAttention = new MultiHeadAttention(dModel, numHeads);
            this.encoderDecoderAttention = new MultiHeadAttention(dModel, numHeads);
            this.feedForward = new PositionwiseFeedForward(dModel, dff);
            this.layerNorm1 = useLayerNorm ? new LayerNorm(dModel) : null;
            this.layerNorm2 = useLayerNorm ? new LayerNorm(dModel) : null;
            this.layerNorm3 = useLayerNorm ? new LayerNorm(dModel) : null;
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

        public void setLayerNorm(LayerNorm layerNorm) {
            this.layerNorm1 = layerNorm;
            this.layerNorm2 = layerNorm;
            this.layerNorm3 = layerNorm;
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
        public INDArray forward(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray cachedInput, INDArray queryPaddingMaskSource, INDArray keyPaddingMaskSource, INDArray queryPaddingMaskTarget, INDArray keyPaddingMaskTarget) {

            // Self-attention avec masque look-ahead
            INDArray attn1 = selfAttention.forwardSelfAttention(x, queryPaddingMaskTarget, lookAheadMask);

            attn1 = dropout1.forward(isTraining, attn1);

            x = x.add(attn1);  // Première connexion résiduelle

            if (layerNorm1 != null) {
                x = layerNorm1.forward(x);
            }


            // Encoder-decoder (cross) attention sans masque look-ahead
            INDArray attn2 = encoderDecoderAttention.forwardCrossAttention(x, encoderOutput, encoderOutput, queryPaddingMaskTarget, keyPaddingMaskSource);

            attn2 = dropout2.forward(isTraining, attn2);

            x = x.add(attn2);  // Deuxième connexion résiduelle

            if (layerNorm2 != null) {
                x = layerNorm2.forward(x);
            }


            // Feed-forward
            INDArray ffOutput = feedForward.forward(x);

            ffOutput = dropout3.forward(isTraining, ffOutput);

            x = x.add(ffOutput);  // Troisième connexion résiduelle

            if (layerNorm3 != null) {
                x = layerNorm3.forward(x);
            }

            return x;
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
            Map<String, INDArray> gradLayerNorm3 = layerNorm3 != null ? layerNorm3.backward(gradOutput.get("input")) : gradOutput;
            gradients.putAll(gradLayerNorm3);

            // Backpropager à travers FeedForward
            Map<String, INDArray> gradFeedForward = feedForward.backward(gradLayerNorm3.get("input"));
            gradients.putAll(gradFeedForward);

            // Backpropager à travers LayerNorm2
            Map<String, INDArray> gradLayerNorm2 = layerNorm2 != null ? layerNorm2.backward(gradFeedForward.get("input")) : gradFeedForward;
            gradients.putAll(gradLayerNorm2);

            // Backpropager à travers EncoderDecoderAttention
            Map<String, INDArray> gradEncoderDecoderAttention = encoderDecoderAttention.backward(gradLayerNorm2.get("input"));
            gradients.putAll(gradEncoderDecoderAttention);

            // Backpropager à travers LayerNorm1
            Map<String, INDArray> gradLayerNorm1 = layerNorm1 != null ? layerNorm1.backward(gradEncoderDecoderAttention.get("input")) : gradEncoderDecoderAttention;
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
            if (layerNorm1 != null) {
                layerParams.addAll(layerNorm1.getParameters());
            }
            if (layerNorm2 != null) {
                layerParams.addAll(layerNorm2.getParameters());
            }
            if (layerNorm3 != null) {
                layerParams.addAll(layerNorm3.getParameters());
            }

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
            if (layerNorm1 != null) {
                layerGrads.addAll(layerNorm1.getGradients());
            }
            if (layerNorm2 != null) {
                layerGrads.addAll(layerNorm2.getGradients());
            }
            if (layerNorm3 != null) {
                layerGrads.addAll(layerNorm3.getGradients());
            }

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
                   (layerNorm1 != null ? layerNorm1.getNumberOfParameters() : 0) +
                   (layerNorm2 != null ? layerNorm2.getNumberOfParameters() : 0) +
                   (layerNorm3 != null ? layerNorm3.getNumberOfParameters() : 0);
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
                   (layerNorm1 != null ? layerNorm1.getNumberOfGradients() : 0) +
                   (layerNorm2 != null ? layerNorm2.getNumberOfGradients() : 0) +
                   (layerNorm3 != null ? layerNorm3.getNumberOfGradients() : 0);
        }

    }

    public void setAttentionDropout(double dropout) {
        this.attentionDropout = dropout;
        // Propager aux couches de décodeur
        for (DecoderLayer layer : layers) {
            layer.dropout1.setDropoutRate(dropout);
            layer.dropout2.setDropoutRate(dropout);
            layer.dropout3.setDropoutRate(dropout);
        }
    }

    /**
     * Met à jour l'échelle de l'encodage positionnel
     * @param scale nouvelle échelle à appliquer
     */
    public void updatePositionalEncodingScale(float scale) {
        this.positionalEncodingScale = scale;
        // Recalculer l'encodage positionnel si nécessaire
        updatePositionalEncoding();
    }

    private void updatePositionalEncoding() {
        if (positionalEncoding != null) {
            positionalEncoding.updateScale(positionalEncodingScale);
        }
    }

    public void setLayerNorm(LayerNorm layerNorm) { 
        this.layerNorm = layerNorm;
    }
}
