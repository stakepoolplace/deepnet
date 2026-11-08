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
         * Passe forward Pre-LN (moderne) : x + SousCouche(LN(x)).
         * Cf. EncoderLayer : résiduels directs, gradients préservés.
         */
        public INDArray forward(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray cachedInput, INDArray queryPaddingMaskSource, INDArray keyPaddingMaskSource, INDArray queryPaddingMaskTarget, INDArray keyPaddingMaskTarget) {

            // Bloc 1 : x + Dropout(SelfAttn(LN1(x))) avec masque look-ahead
            INDArray nx = layerNorm1 != null ? layerNorm1.forward(x) : x;
            INDArray attn1 = selfAttention.forwardSelfAttention(nx, queryPaddingMaskTarget, lookAheadMask);

            attn1 = dropout1.forward(isTraining, attn1);

            INDArray h1 = x.add(attn1);  // Première connexion résiduelle directe


            // Bloc 2 : h1 + Dropout(CrossAttn(LN2(h1))) sans masque look-ahead
            INDArray nh1 = layerNorm2 != null ? layerNorm2.forward(h1) : h1;
            INDArray attn2 = encoderDecoderAttention.forwardCrossAttention(nh1, encoderOutput, encoderOutput, queryPaddingMaskTarget, keyPaddingMaskSource);

            attn2 = dropout2.forward(isTraining, attn2);

            INDArray h2 = h1.add(attn2);  // Deuxième connexion résiduelle directe


            // Bloc 3 : h2 + Dropout(FF(LN3(h2)))
            INDArray nh2 = layerNorm3 != null ? layerNorm3.forward(h2) : h2;
            INDArray ffOutput = feedForward.forward(nh2);

            ffOutput = dropout3.forward(isTraining, ffOutput);

            INDArray out = h2.add(ffOutput);  // Troisième connexion résiduelle directe

            return out;
        }

        /**
         * Passe backward Pre-LN (miroir du forward ci-dessus).
         *
         * @param gradOutput Gradient provenant de la couche suivante.
         * @param cachedInput Entrée mise en cache de la passe forward précédente (inutilisée).
         * @return Gradient à propager vers la couche précédente.
         */
        public Map<String, INDArray> backward(Map<String, INDArray> gradOutput, INDArray cachedInput) {
            Map<String, INDArray> gradients = new HashMap<>();

            // dL/d(sortie de couche)
            INDArray gOut = gradOutput.get("input");

            // Bloc 3 : out = h2 + drop(FF(LN3(h2)))
            INDArray dFF = dropout3.backward(gOut); // dL/d(FF(LN3(h2)))
            Map<String, INDArray> gradFeedForward = feedForward.backward(dFF); // dL/d(LN3(h2))
            gradients.putAll(gradFeedForward);
            INDArray dNH2 = gradFeedForward.get("input");
            Map<String, INDArray> gradLayerNorm3 = layerNorm3 != null
                    ? layerNorm3.backward(dNH2) : gradOutput;
            gradients.putAll(gradLayerNorm3);
            // LN nulle (useLayerNorm=false) : pas de normalisation, la voie FF
            // est dNH2 elle-même (et non le "input" de gradOutput qui vaut gOut).
            INDArray dH2ViaFF = layerNorm3 != null ? gradLayerNorm3.get("input") : dNH2; // dL/d(h2) via FF
            // Connexion résiduelle 3 : dL/dh2 = skip + voie FF
            INDArray dh2 = gOut.add(dH2ViaFF);

            // Bloc 2 : h2 = h1 + drop(CrossAttn(LN2(h1)))
            Map<String, INDArray> gradEncoderDecoderAttention = encoderDecoderAttention.backward(dropout2.backward(dh2));
            // Sauvegarder les gradients vers l'encodeur AVANT tout écrasement :
            // l'encodeur alimente K et V, son gradient est gradInputK + gradInputV
            INDArray gradToEncoderK = gradEncoderDecoderAttention.get("gradInputK");
            INDArray gradToEncoderV = gradEncoderDecoderAttention.get("gradInputV");
            gradients.putAll(gradEncoderDecoderAttention);

            // dL/d(LN2(h1)) = dL/dQ (côté requête de la cross-attention)
            INDArray dNH1 = gradEncoderDecoderAttention.get("gradInputQ");
            Map<String, INDArray> gradLayerNorm2 = layerNorm2 != null
                    ? layerNorm2.backward(dNH1) : gradOutput;
            gradients.putAll(gradLayerNorm2);
            INDArray dH1ViaCross = layerNorm2 != null ? gradLayerNorm2.get("input") : dNH1; // dL/d(h1) via cross
            // Connexion résiduelle 2 : dL/dh1 = skip + voie cross
            INDArray dh1 = dh2.add(dH1ViaCross);

            // Bloc 1 : h1 = x + drop(SelfAttn(LN1(x)))) ; x alimente Q, K et V
            Map<String, INDArray> gradSelfAttention = selfAttention.backward(dropout1.backward(dh1));
            INDArray dNx = gradSelfAttention.get("gradInputQ")
                    .add(gradSelfAttention.get("gradInputK"))
                    .add(gradSelfAttention.get("gradInputV"));
            Map<String, INDArray> gradLayerNorm1 = layerNorm1 != null
                    ? layerNorm1.backward(dNx) : gradOutput;
            gradients.putAll(gradLayerNorm1);
            INDArray dXViaSelf = layerNorm1 != null ? gradLayerNorm1.get("input") : dNx; // dL/dx via self-attention
            // Connexion résiduelle 1 : dL/dx = skip + voie self-attention
            INDArray dx = dh1.add(dXViaSelf);
            gradients.putAll(gradSelfAttention);

            // Restaurer les gradients vers l'encodeur (écrasés par la self-attention
            // qui utilise les mêmes clés) et chaîner le gradient d'entrée
            gradients.put("gradInputK", gradToEncoderK);
            gradients.put("gradInputV", gradToEncoderV);
            gradients.put("input", dx);

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
