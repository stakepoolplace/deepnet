package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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


    public Decoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dropoutRate = dropoutRate;
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.linearProjection = new LinearProjection(dModel, TransformerModel.getVocabSize()); // Initialiser avec la taille du vocabulaire
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

    public INDArray decode(boolean isTraining, INDArray encoderOutput, INDArray encodedDecoderInput, INDArray lookAheadMask, INDArray paddingMask) {
        // Réinitialiser le cache avant une nouvelle passe forward
        resetCache();
    
        // Traitement par les couches de décodeur
        for (int i = 0; i < layers.size(); i++) {
            DecoderLayer layer = layers.get(i);
            encodedDecoderInput = layer.forward(isTraining, encodedDecoderInput, encoderOutput, lookAheadMask, paddingMask, forwardCache.get(i));
            forwardCache.set(i, encodedDecoderInput.dup()); // Stocker l'entrée actuelle dans le cache
        }
    
        // Normalisation finale
        encodedDecoderInput = layerNorm.forward(encodedDecoderInput);
        lastNormalizedInput = encodedDecoderInput.dup(); // Stocker l'entrée normalisée
    
        // Projection linéaire vers le vocabulaire
        INDArray logits = linearProjection.project(encodedDecoderInput); // [batchSize, targetSeqLength, vocabSize]
        System.out.println("Logits shape: " + Arrays.toString(logits.shape()));
    
        return logits;
    }
    
    public Map<String, INDArray> backward(INDArray gradOutput) {
        // Récupérer l'entrée normalisée de la passe forward
        if (lastNormalizedInput == null) {
            throw new IllegalStateException("L'entrée normalisée n'est pas initialisée. Assurez-vous d'effectuer une passe forward avant.");
        }
    
        // Propager le gradient à travers LinearProjection
        Map<String, INDArray> gradLinearProjection = linearProjection.backward(lastNormalizedInput, gradOutput);
    
        // Propager le gradient à travers LayerNorm
        Map<String, INDArray> gradLayerNorm = layerNorm.backward(gradLinearProjection.get("input"));
    
        // Transformer gradLayerNorm en un Map pour correspondre à la signature de DecoderLayer.backward
        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("input", gradLayerNorm.get("input")); // Utiliser "input" comme clé est arbitraire mais doit correspondre à ce que s'attend à recevoir DecoderLayer.backward
    
        // Commencer avec le gradient à la sortie du Decoder
        for (int i = layers.size() - 1; i >= 0; i--) {
            DecoderLayer layer = layers.get(i);
            // Passer le cache correspondant à cette couche
            INDArray layerInput = forwardCache.get(i);
            gradMap = layer.backward(gradMap, layerInput);
        }
        // À ce stade, gradMap contiendrait le gradient à propager à l'Encoder
        // Vous pouvez ensuite extraire le gradient à passer à l'encodeur ou à d'autres parties du modèle si nécessaire.
    
        return gradMap;
    }
    

    // Méthode pour obtenir tous les paramètres du décodeur
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

    // Méthode pour obtenir tous les gradients du décodeur
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

    // Méthode pour calculer les gradients basés sur la perte
    public INDArray calculateGradients(double loss) {
        // La logique réelle de calcul des gradients serait beaucoup plus complexe
        // et dépendrait des détails spécifiques de votre implémentation et de votre bibliothèque d'autograd.
        INDArray gradients = Nd4j.rand(1, 100); // Assumer des dimensions hypothétiques pour l'exemple
        return gradients;
    }

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

        /**
         * Passe forward avec gestion du cache.
         *
         * @param isTraining       Indique si le modèle est en mode entraînement
         * @param x                Entrée actuelle
         * @param encoderOutput    Sortie de l'encodeur
         * @param lookAheadMask    Masque de look-ahead
         * @param paddingMask      Masque de padding
         * @param cachedInput      Entrée mise en cache de la passe forward précédente
         * @return Sortie après cette couche
         */
        public INDArray forward(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask, INDArray cachedInput) {
            
            // Vérification des formes
            if (x.rank() != 3 || encoderOutput.rank() != 3) {
                throw new IllegalArgumentException("Les entrées query, key et value doivent être de rang 3.");
            }
            
            INDArray attn1 = selfAttention.forward(x, x, x, lookAheadMask);
            attn1 = dropout1.apply(isTraining, attn1);
            x = layerNorm1.forward(x.add(attn1));

            INDArray attn2 = encoderDecoderAttention.forward(x, encoderOutput, encoderOutput, paddingMask);
            attn2 = dropout2.apply(isTraining, attn2);
            x = layerNorm2.forward(x.add(attn2));

            INDArray ffOutput = feedForward.forward(x);
            ffOutput = dropout3.apply(isTraining, ffOutput);
            return layerNorm3.forward(x.add(ffOutput));
        }

        /**
         * Passe backward avec utilisation du cache.
         *
         * @param gradOutput Gradients provenant de la couche suivante
         * @param cachedInput Entrée mise en cache de la passe forward précédente
         * @return Gradients à propager vers les couches précédentes
         */
        public Map<String, INDArray> backward(Map<String, INDArray> gradOutput, INDArray cachedInput) {
            
            Map<String, INDArray> gradients = new HashMap<>();

        	// Rétropropagation à travers LayerNorm3
            Map<String, INDArray> gradLayerNorm3 = layerNorm3.backward(gradOutput.get("input"));
            gradients.putAll(gradLayerNorm3);

            // Rétropropagation à travers PositionwiseFeedForward
            Map<String, INDArray> gradFeedForward = feedForward.backward(gradLayerNorm3.get("input"));
            gradients.putAll(gradFeedForward);

            // Rétropropagation à travers LayerNorm2
            Map<String, INDArray> gradLayerNorm2 = layerNorm2.backward(gradFeedForward.get("input"));
            gradients.putAll(gradLayerNorm2);

            // Rétropropagation à travers encoderDecoderAttention
            Map<String, INDArray> gradEncoderDecoderAttention = encoderDecoderAttention.backward(gradLayerNorm2.get("input"));
            gradients.putAll(gradEncoderDecoderAttention);

            // Rétropropagation à travers LayerNorm1
            Map<String, INDArray> gradLayerNorm1 = layerNorm1.backward(gradEncoderDecoderAttention.get("input"));
            gradients.putAll(gradLayerNorm1);

            // Rétropropagation à travers selfAttention
            Map<String, INDArray> gradSelfAttention = selfAttention.backward(gradLayerNorm1.get("input"));
            gradients.putAll(gradSelfAttention);

            // Retourner les gradients accumulés pour mise à jour des paramètres
            return gradients;
        }

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

        public long getNumberOfParameters() {
            return selfAttention.getNumberOfParameters() +
                   encoderDecoderAttention.getNumberOfParameters() +
                   feedForward.getNumberOfParameters() +
                   layerNorm1.getNumberOfParameters() +
                   layerNorm2.getNumberOfParameters() +
                   layerNorm3.getNumberOfParameters();
        }

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
