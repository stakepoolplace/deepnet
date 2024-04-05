package RN.transformer;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    private int dModel;
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
    
    public List<INDArray> decode(INDArray encodedData) {
        // Applique une projection linéaire à l'ensemble du batch d'encodages pour obtenir les logits
        INDArray logits = linearProjection.project(encodedData);

        // Emballez les logits dans une liste pour maintenir la cohérence avec l'interface attendue
        List<INDArray> decodedLogits = new ArrayList<>();
        decodedLogits.add(logits); // Ajoutez directement les logits comme un seul élément de la liste

        return decodedLogits;
    }


    public INDArray forward(INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask) {
        for (DecoderLayer layer : layers) {
            x = layer.forward(x, encoderOutput, lookAheadMask, paddingMask);
        }
        x = layerNorm.forward(x);
        x = linearProjection.project(x); // Applique une projection linéaire à la sortie du décodeur
        return x;
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
        
        public INDArray forward(INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask) {
            INDArray attn1 = selfAttention.forward(x, x, x, lookAheadMask);
            attn1 = dropout1.apply(attn1);
            x = layerNorm1.forward(x.add(attn1));

            INDArray attn2 = encoderDecoderAttention.forward(x, encoderOutput, encoderOutput, paddingMask);
            attn2 = dropout2.apply(attn2);
            x = layerNorm2.forward(x.add(attn2));

            INDArray ffOutput = feedForward.forward(x);
            ffOutput = dropout3.apply(ffOutput);
            return layerNorm3.forward(x.add(ffOutput));
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
        
        public long getNumberOfParameters() {
            return selfAttention.getNumberOfParameters() +
                   encoderDecoderAttention.getNumberOfParameters() +
                   feedForward.getNumberOfParameters() +
                   layerNorm1.getNumberOfParameters() +
                   layerNorm2.getNumberOfParameters() +
                   layerNorm3.getNumberOfParameters();
        }


    }
}
