package RN.transformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import RN.transformer.Encoder.EncoderLayer;

/**
 * numLayers: Comme pour l'encodeur.
 * dModel: Identique à celui de l'encodeur.
 * numHeads: Identique à celui de l'encodeur.
 * dff: Identique à celui de l'encodeur.
 * vocabSize: Identique à celui de l'encodeur, supposant un vocabulaire partagé entre l'encodeur et le décodeur.
 * maxSeqLength: Identique à celui de l'encodeur.
 */
public class Decoder {
    private List<DecoderLayer> layers;
    private LayerNorm layerNorm;
    private int numLayers;
    protected int dModel;
    private int numHeads;
    private double dropoutRate;
    private LinearProjection linearProjection; // Projection linéaire vers la taille du vocabulaire

    public Decoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, int vocabSize) {
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dropoutRate = dropoutRate;
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.linearProjection = new LinearProjection(dModel, vocabSize); // Initialiser avec la taille du vocabulaire

        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new DecoderLayer(dModel, numHeads, dff, dropoutRate));
        }
    }
    

    
    public INDArray decode(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask) {
        // Traitement par les couches de décodeur
        for (DecoderLayer layer : layers) {
            x = layer.forward(isTraining, x, encoderOutput, lookAheadMask, paddingMask);
        }
        
        // Normalisation finale
        x = layerNorm.forward(x);
        
        // Application d'une projection linéaire, si nécessaire
        if (this.linearProjection != null) {
            x = linearProjection.project(x); // Transforme la sortie du décodeur pour les logits de vocabulaire
        }
        
        return x;
    }
    
    
    public Map<String, INDArray> backward(INDArray gradOutput) {
        
        // D'abord, propager le gradient à travers LinearProjection
        Map<String, INDArray> gradLinearProjection = linearProjection.backward(gradOutput);

        // Ensuite, propager le gradient à travers LayerNorm
        // Supposons que backward de LayerNorm attend et retourne un INDArray pour simplifier
        Map<String, INDArray> gradLayerNorm = layerNorm.backward(gradLinearProjection.get("input"));

    	
    	// Transformer gradOutput en un Map pour correspondre à la signature de DecoderLayer.backward
        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("input", gradLayerNorm.get("input")); // Utiliser "input" comme clé est arbitraire mais doit correspondre à ce que s'attend à recevoir DecoderLayer.backward
        
        // Commencer avec le gradient à la sortie du Decoder
        for (int i = layers.size() - 1; i >= 0; i--) {
            DecoderLayer layer = layers.get(i);
            gradMap = layer.backward(gradMap); // Ici, gradMap est attendu et retourné par backward
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

    static class DecoderLayer {
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
        
        public INDArray forward(boolean isTraining, INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask) {
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
        
        public Map<String, INDArray> backward(Map<String, INDArray> gradOutput) {
            
            // Rétropropagation à travers LayerNorm3
            Map<String, INDArray> gradLayerNorm3 = layerNorm3.backward(gradOutput.get("input"));
            
            // Rétropropagation à travers PositionwiseFeedForward
            Map<String, INDArray> gradFeedForward = feedForward.backward(gradLayerNorm3.get("input"));
            
            // Rétropropagation à travers LayerNorm2
            Map<String, INDArray> gradLayerNorm2 = layerNorm2.backward(gradFeedForward.get("input"));
            
            // Rétropropagation à travers encoderDecoderAttention
            Map<String, INDArray> gradEncoderDecoderAttention = encoderDecoderAttention.backward(gradLayerNorm2.get("input"));
            
            // Rétropropagation à travers LayerNorm1
            Map<String, INDArray> gradLayerNorm1 = layerNorm1.backward(gradEncoderDecoderAttention.get("inputQ"));
            
            // Rétropropagation à travers selfAttention
            Map<String, INDArray> gradSelfAttention = selfAttention.backward(gradLayerNorm1.get("input"));
            
            // Si nécessaire, propager le gradient à travers d'autres opérations ou couches...
            
            return gradSelfAttention; // Retourner les gradients accumulés pour mise à jour des paramètres
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
