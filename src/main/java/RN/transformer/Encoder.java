// Encoder.java
package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import RN.utils.NDArrayUtils;

/**
 * Classe représentant l'encodeur du modèle Transformer.
 */
public class Encoder implements Serializable {
    
    private static final long serialVersionUID = -5716799542280937448L;
    List<EncoderLayer> layers;
    private int dModel;
    private PositionalEncoding positionalEncoding;
    private LayerNorm layerNorm;
    private Tokenizer tokenizer;
    private double attentionDropout = 0.0;
    private float positionalEncodingScale = 1.0f;

    public Encoder() {
    }
    
    public Encoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, Tokenizer tokenizer, boolean useLayerNorm) {
        this.dModel = dModel;
        this.positionalEncoding = new PositionalEncoding(dModel);
        this.layers = new ArrayList<>();
        this.layerNorm = useLayerNorm ? new LayerNorm(dModel) : null;
        this.tokenizer = tokenizer;
        
        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new EncoderLayer(this, dModel, numHeads, dff, dropoutRate, useLayerNorm));
        }
    }

    /**
     * Encode un batch de séquences d'IDs de tokens.
     *
     * @param isTraining    Indique si le modèle est en mode entraînement.
     * @param data           INDArray représentant les IDs de tokens des séquences [batchSize, seqLength].
     * @param paddingMask    Masque de padding pour le batch [batchSize, 1, 1, seqLength].
     * @return Représentations encodées [batchSize, seqLength, dModel].
     */
    public INDArray encode(boolean isTraining, Batch batch) {

        INDArray data = batch.getData();

        // Vérification des dimensions
        if (data.rank() != 2) {
            throw new IllegalArgumentException("Data doit être de rang 2 [batchSize, seqLength], mais a la forme: " + java.util.Arrays.toString(data.shape()));
        }

        INDArray keyPaddingMask = NDArrayUtils.createKeyPaddingMask(tokenizer, data);
        INDArray queryPaddingMask = NDArrayUtils.createQueryPaddingMask(tokenizer, data);

        // Lookup embeddings: [batchSize, seqLength, dModel]
        INDArray inputEmbeddings = tokenizer.lookupEmbeddings(data); 

        // Appliquer les encodages positionnels
        INDArray posEncoding = positionalEncoding.getPositionalEncoding(data.shape()[1]); // [seqLength, dModel]
        
        // Étendre le posEncoding pour qu'il soit compatible avec le batch
        posEncoding = posEncoding.reshape(1, data.shape()[1], dModel).broadcast(inputEmbeddings.shape());

        INDArray x = inputEmbeddings.add(posEncoding); // [batchSize, seqLength, dModel]
        // System.out.println("After positional encoding: " + java.util.Arrays.toString(x.shape()));
    
        // Passer à travers les couches de l'encodeur
        for (EncoderLayer layer : layers) {
            x = layer.forward(isTraining, x, queryPaddingMask, keyPaddingMask);
            // System.out.println("After encoder layer: " + java.util.Arrays.toString(x.shape()));
        }
        
        // Appliquer la normalisation finale
        x = layerNorm != null ? layerNorm.forward(x) : x;
        // System.out.println("After layer normalization: " + java.util.Arrays.toString(x.shape()));
    
        return x;
    }

 
    /**
     * Passe backward à travers l'encodeur.
     *
     * @param gradOutput Gradient provenant de la couche suivante.
     */
    public Map<String, INDArray> backward(Map<String, INDArray> gradOutput) {
        // Récupérer gradAttentionOutputConcat
        INDArray gradAttentionOutputConcatND = gradOutput.get("gradAttentionOutputConcat"); // [batchSize, numHeads, seqLength, depth]

        if (gradAttentionOutputConcatND == null) {
            throw new IllegalStateException("gradAttentionOutputConcat est null. Assurez-vous que MultiHeadAttention.backward retourne correctement ce gradient.");
        }

        // Permuter les axes [batchSize, numHeads, seqLength, depth] en [batchSize, seqLength, numHeads, depth]
        INDArray gradPermuted = gradAttentionOutputConcatND.permute(0, 2, 1, 3); // [batchSize, seqLength, numHeads, depth]

        // Calculer dynamiquement les dimensions
        int batchSize = (int) gradPermuted.size(0);
        int seqLength = (int) gradPermuted.size(1);
        int numHeads = layers.get(0).getNumHeads();
        int depth = dModel / numHeads;

        // Vérifier que dModel est divisible par numHeads
        if (dModel != numHeads * depth) {
            throw new IllegalStateException("dModel doit être divisible par numHeads. dModel=" + dModel + ", numHeads=" + numHeads);
        }

        // Reshaper en [batchSize, seqLength, numHeads * depth] == [batchSize, seqLength, dModel]
        INDArray gradOutputForLayerNorm = gradPermuted.reshape(batchSize, seqLength, numHeads * depth); // [batchSize, seqLength, dModel]

        // Backpropager à travers la normalisation de couche finale
        Map<String, INDArray> gradientsFromLayerNorm = layerNorm != null ? layerNorm.backward(gradOutputForLayerNorm) : new HashMap<>();

        // Obtenir le gradient à passer aux couches de l'encodeur
        INDArray gradInput = gradientsFromLayerNorm.get("input");

        // Backpropager à travers chaque couche d'encodeur dans l'ordre inverse
        for (int i = layers.size() - 1; i >= 0; i--) {
            EncoderLayer layer = layers.get(i);
            gradInput = layer.backward(gradInput);
            if (gradInput == null) {
                throw new IllegalArgumentException("gradInput est null après backward de la couche " + i);
            }
        }

        // Maintenant, gradInput est le gradient par rapport aux embeddings
        Map<String, INDArray> gradients = new HashMap<>();
        gradients.put("gradEmbeddings", gradInput);

        return gradients;
    }



    /**
     * Obtient tous les paramètres de l'encodeur.
     *
     * @return Liste des paramètres.
     */
    public List<INDArray> getParameters() {
        List<INDArray> params = new ArrayList<>();
        // Collecter les paramètres de toutes les couches de l'encodeur
        for (EncoderLayer layer : layers) {
            params.addAll(layer.getParameters());
        }

        // Inclure les paramètres de la normalisation de couche
        if(layerNorm != null) {
            params.addAll(layerNorm.getParameters());
        }
        
        return params;
    }

    /**
     * Obtient tous les gradients de l'encodeur.
     *
     * @return Liste des gradients.
     */
    public List<INDArray> getGradients() {
        List<INDArray> grads = new ArrayList<>();
        // Collecter les gradients de toutes les couches de l'encodeur
        for (EncoderLayer layer : layers) {
            grads.addAll(layer.getGradients());
        }

        // Inclure les gradients de la normalisation de couche
        if(layerNorm != null) {
            grads.addAll(layerNorm.getGradients());
        }
        
        return grads;
    }

    /**
     * Obtient le nombre total de paramètres dans l'encodeur.
     *
     * @return Nombre de paramètres.
     */
    public int getNumberOfParameters() {
        int numParams = 0;

        // Parcourir toutes les couches d'encodeur pour compter leurs paramètres
        for (EncoderLayer layer : layers) {
            numParams += layer.getNumberOfParameters();
        }

        // Ajouter les paramètres de la normalisation de couche
        numParams += layerNorm != null ? layerNorm.getNumberOfParameters() : 0;

        return numParams;
    }

    /**
     * Obtient le nombre total de gradients dans l'encodeur.
     *
     * @return Nombre de gradients.
     */
    public int getNumberOfGradients() {
        int numGrads = 0;

        // Parcourir toutes les couches d'encodeur pour compter leurs gradients
        for (EncoderLayer layer : layers) {
            numGrads += layer.getNumberOfGradients();
        }

        // Ajouter les gradients de la normalisation de couche
        numGrads += layerNorm != null ? layerNorm.getNumberOfGradients() : 0;

        return numGrads;
    }

    /**
     * Classe interne représentant une couche unique de l'encodeur.
     */
    static class EncoderLayer implements Serializable {
        
        private static final long serialVersionUID = -88886021425567141L;
        
        MultiHeadAttention selfAttention;
        PositionwiseFeedForward feedForward;
        LayerNorm layerNorm1;
        LayerNorm layerNorm2;
        Dropout dropout1;
        Dropout dropout2;
        Encoder encoder;

        public EncoderLayer(Encoder encoder, int dModel, int numHeads, int dff, double dropoutRate, boolean useLayerNorm) {
            this.encoder = encoder;
            this.selfAttention = new MultiHeadAttention(dModel, numHeads);
            this.feedForward = new PositionwiseFeedForward(dModel, dff);
            this.layerNorm1 = useLayerNorm ? new LayerNorm(dModel) : null;
            this.layerNorm2 = useLayerNorm ? new LayerNorm(dModel) : null;
            this.dropout1 = new Dropout(dropoutRate);
            this.dropout2 = new Dropout(dropoutRate);
        }

        public MultiHeadAttention getSelfAttention() {
            return this.selfAttention;
        }
        
        /**
         * Passe forward à travers la couche d'encodeur.
         *
         * @param isTraining  Indique si le modèle est en mode entraînement.
         * @param x           Entrée de la couche [batchSize, seqLength, dModel].
         * @param paddingMask Masque de padding [batchSize, 1, 1, seqLength].
         * @return Sortie de la couche [batchSize, seqLength, dModel].
         */
        public INDArray forward(boolean isTraining, INDArray x, INDArray queryPaddingMask, INDArray keyPaddingMask) {
            
         
            
            if (keyPaddingMask.rank() != 4) {
                throw new IllegalArgumentException("keyPaddingMask doit être de rang 4 [batchSize, 1, 1, seqLength], mais a la forme: " + java.util.Arrays.toString(keyPaddingMask.shape()));
            }

            // Attention multi-têtes auto-attention
            INDArray attnOutput = selfAttention.forwardSelfAttention(x, queryPaddingMask, null);

            attnOutput = dropout1.forward(isTraining, attnOutput); // Appliquer dropout
            INDArray out1 = layerNorm1 != null ? layerNorm1.forward(x.add(attnOutput)) : x.add(attnOutput); // Add & Norm [batchSize, seqLength, dModel]

            // Feed-forward
            INDArray ffOutput = feedForward.forward(out1); // [batchSize, seqLength, dModel]
            ffOutput = dropout2.forward(isTraining, ffOutput); // Appliquer dropout
            INDArray out2 = layerNorm2 != null ? layerNorm2.forward(out1.add(ffOutput)) : out1.add(ffOutput); // Add & Norm [batchSize, seqLength, dModel]

            return out2;
        }
        
        /**
         * Passe backward à travers la couche d'encodeur.
         *
         * @param gradOutput Gradient provenant de la couche suivante [batchSize, seqLength, dModel].
         * @return Gradient à propager vers la couche précédente [batchSize, seqLength, dModel].
         */
        public INDArray backward(INDArray gradOutput) {
            // Backward à travers la deuxième normalisation de couche
            Map<String, INDArray> gradLayerNorm2 = layerNorm2 != null ? layerNorm2.backward(gradOutput) : new HashMap<>(); // {'input': grad_out1}
            INDArray gradAddNorm2 = gradLayerNorm2.get("input"); // [batchSize, seqLength, dModel]

            // Backward à travers le deuxième Add & Norm (FeedForward)
            Map<String, INDArray> gradFeedForwardMap = feedForward.backward(gradAddNorm2); // {'input': grad_ff}
            INDArray gradFeedForward = gradFeedForwardMap.get("input"); // [batchSize, seqLength, dModel]
            gradFeedForward = dropout2.backward(gradFeedForward); // [batchSize, seqLength, dModel]


            // Gradient après FeedForward
            INDArray gradOut1 = gradAddNorm2.add(gradFeedForward); // [batchSize, seqLength, dModel]

            // Backward à travers la première normalisation de couche
            Map<String, INDArray> gradLayerNorm1 = layerNorm1 != null ? layerNorm1.backward(gradOut1) : new HashMap<>(); // {'input': grad_attn}
            INDArray gradAddNorm1 = gradLayerNorm1.get("input"); // [batchSize, seqLength, dModel]

            // Backward à travers le premier Add & Norm (Self-Attention)
            Map<String, INDArray> gradSelfAttentionMap = selfAttention.backward(gradAddNorm1); // {'input': grad_attn}
            INDArray gradSelfAttention = gradSelfAttentionMap.get("input"); // [batchSize, seqLength, dModel]
            gradSelfAttention = dropout1.backward(gradSelfAttention); // [batchSize, seqLength, dModel]

            // Gradient à propager vers les couches précédentes
            return gradSelfAttention;
        }


        /**
         * Obtient tous les paramètres de la couche d'encodeur.
         *
         * @return Liste des paramètres.
         */
        public List<INDArray> getParameters() {
            List<INDArray> layerParams = new ArrayList<>();
            
            layerParams.addAll(selfAttention.getParameters());
            layerParams.addAll(feedForward.getParameters());
            layerParams.addAll(layerNorm1 != null ? layerNorm1.getParameters() : new ArrayList<>());
            layerParams.addAll(layerNorm2 != null ? layerNorm2.getParameters() : new ArrayList<>());

            return layerParams;
        }

        /**
         * Obtient tous les gradients de la couche d'encodeur.
         *
         * @return Liste des gradients.
         */
        public List<INDArray> getGradients() {
            List<INDArray> layerGrads = new ArrayList<>();
            
            layerGrads.addAll(selfAttention.getGradients());
            layerGrads.addAll(feedForward.getGradients());
            layerGrads.addAll(layerNorm1 != null ? layerNorm1.getGradients() : new ArrayList<>());
            layerGrads.addAll(layerNorm2 != null ? layerNorm2.getGradients() : new ArrayList<>());

            return layerGrads;
        }

        /**
         * Obtient le nombre total de paramètres dans la couche d'encodeur.
         *
         * @return Nombre de paramètres.
         */
        public long getNumberOfParameters() {
            return selfAttention.getNumberOfParameters() +
                   feedForward.getNumberOfParameters() +
                   (layerNorm1 != null ? layerNorm1.getNumberOfParameters() : 0) +
                   (layerNorm2 != null ? layerNorm2.getNumberOfParameters() : 0);
        }

        /**
         * Obtient le nombre total de gradients dans la couche d'encodeur.
         *
         * @return Nombre de gradients.
         */
        public long getNumberOfGradients() {
            return selfAttention.getNumberOfGradients() +
                   feedForward.getNumberOfGradients() +
                   (layerNorm1 != null ? layerNorm1.getNumberOfGradients() : 0) +
                   (layerNorm2 != null ? layerNorm2.getNumberOfGradients() : 0);
        }

        /**
         * Getter pour le nombre de têtes d'attention dans cette couche.
         *
         * @return Nombre de têtes d'attention.
         */
        public int getNumHeads() {
            return selfAttention.getNumHeads();
        }
    }

    public void setAttentionDropout(double dropout) {
        this.attentionDropout = dropout;
        // Propager aux couches d'encodeur
        for (EncoderLayer layer : layers) {
            layer.dropout1.setDropoutRate(dropout);
            layer.dropout2.setDropoutRate(dropout);
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
}
