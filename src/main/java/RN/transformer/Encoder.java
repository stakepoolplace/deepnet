package RN.transformer;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.ArrayList;
import java.util.List;


/**
 * 	numLayers: Le nombre de couches répétitives dans l'encodeur.
 *	dModel: La dimensionnalité des embeddings de tokens et des sorties de toutes les couches dans le modèle.
 *	numHeads: Le nombre de têtes d'attention dans les mécanismes d'attention multi-têtes.
 *	dff: La dimensionnalité des couches feed-forward internes dans chaque couche d'encodeur.
 *	vocabSize: La taille du vocabulaire, nécessaire pour les embeddings de tokens.
 *	maxSeqLength: La longueur maximale de séquence, utilisée pour les embeddings positionnels.
 */

public class Encoder {
	
    private List<EncoderLayer> layers;
    private PositionalEncoding positionalEncoding;
    private LayerNorm layerNorm;
    private INDArray pretrainedEmbeddings; // Matrice d'embeddings pré-entraînée
    private Tokenizer tokenizer;

    public Encoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, INDArray pretrainedEmbeddings, Tokenizer tokenizer) {
        this.positionalEncoding = new PositionalEncoding(dModel);
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.pretrainedEmbeddings = pretrainedEmbeddings; // Initialiser la matrice d'embeddings pré-entraînée
        this.tokenizer = tokenizer;
        
        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new EncoderLayer(dModel, numHeads, dff, dropoutRate));
        }
    }

    public List<List<Float>> encode(List<String> words) {
        // Tokenization du texte
        List<String> tokens = tokenizer.tokenize(words);
        // Conversion des tokens en IDs
        List<Integer> tokenIds = tokenizer.tokensToIds(tokens);

        // Encodage des IDs de tokens à travers les couches de l'encodeur
        INDArray inputEmbeddings = lookupEmbeddings(tokenIds);
        INDArray encoded = forward(inputEmbeddings);
        
        // Conversion des embeddings encodés en logits
        return convertToLogits(encoded);
    }

    private INDArray lookupEmbeddings(List<Integer> tokenIds) {
        // Utiliser la matrice d'embeddings pré-entraînée pour récupérer les embeddings correspondants aux IDs de tokens
        int maxSeqLength = tokenIds.size();
        int dModel = layers.get(0).selfAttention.getdModel();
        INDArray embeddings = Nd4j.zeros(maxSeqLength, dModel);

        for (int i = 0; i < tokenIds.size(); i++) {
            int tokenId = tokenIds.get(i);
            // Récupérer l'embedding correspondant au token ID à partir de la matrice d'embeddings pré-entraînée
            embeddings.putRow(i, pretrainedEmbeddings.getRow(tokenId));
        }

        return embeddings;
    }

    private List<List<Float>> convertToLogits(INDArray encoded) {
        // Convertir les embeddings encodés en logits
        List<List<Float>> logits = new ArrayList<>();
        int seqLength = (int) encoded.size(0);
        for (int i = 0; i < seqLength; i++) {
            INDArray row = encoded.getRow(i);
            List<Float> rowList = new ArrayList<>();
            for (int j = 0; j < row.length(); j++) {
                rowList.add(row.getFloat(j)); // Ajouter la valeur de l'élément à la liste des logits
            }
            logits.add(rowList);
        }
        return logits;
    }

    private INDArray forward(INDArray x) {
        // Appliquer les embeddings positionnels
        INDArray posEncoding = positionalEncoding.getPositionalEncoding(x.shape()[0]);
        x = x.add(posEncoding);

        for (EncoderLayer layer : layers) {
            x = layer.forward(x);
        }
        
        return layerNorm.forward(x);
    }
    
    
    public List<INDArray> getParameters() {
        List<INDArray> params = new ArrayList<>();
        // Collecter les poids et biais de multiHeadAttention et positionwiseFeedForward
        for (EncoderLayer layer : layers) {
            params.addAll(layer.getParameters());
        }

        // Inclure les paramètres de la normalisation de couche finale
        if(layerNorm != null) {
            params.addAll(layerNorm.getParameters());
        }
        
        return params;
    }

    // Méthode pour calculer les gradients basés sur la perte
    public INDArray calculateGradients(double loss) {
        // Dans un cas réel, cette méthode impliquerait le calcul du gradient de la perte par rapport à chaque paramètre
        // Pour cet exemple, simuler un gradient comme un INDArray de mêmes dimensions que les paramètres
        INDArray gradients = Nd4j.rand(1, 100); // Assumer les mêmes dimensions hypothétiques que les paramètres
        return gradients;
    }

    static class EncoderLayer {
        MultiHeadAttention selfAttention;
        PositionwiseFeedForward feedForward;
        LayerNorm layerNorm1;
        LayerNorm layerNorm2;
        Dropout dropout1;
        Dropout dropout2;

        public EncoderLayer(int dModel, int numHeads, int dff, double dropoutRate) {
            this.selfAttention = new MultiHeadAttention(dModel, numHeads);
            this.feedForward = new PositionwiseFeedForward(dModel, dff);
            this.layerNorm1 = new LayerNorm(dModel);
            this.layerNorm2 = new LayerNorm(dModel);
            this.dropout1 = new Dropout(dropoutRate);
            this.dropout2 = new Dropout(dropoutRate);
        }
        
        public List<INDArray> getParameters() {
            List<INDArray> layerParams = new ArrayList<>();
            
            // Collecter les paramètres des composants de la couche d'encodeur
            layerParams.addAll(selfAttention.getParameters());
            layerParams.addAll(feedForward.getParameters());
            layerParams.addAll(layerNorm1.getParameters());
            layerParams.addAll(layerNorm2.getParameters());

            return layerParams;
        }

        public INDArray forward(INDArray x) {
            INDArray attnOutput = selfAttention.forward(x, x, x, null); // Assume no need for mask here
            attnOutput = dropout1.apply(attnOutput);
            x = layerNorm1.forward(x.add(attnOutput)); // Add & norm

            INDArray ffOutput = feedForward.forward(x);
            ffOutput = dropout2.apply(ffOutput);
            return layerNorm2.forward(x.add(ffOutput)); // Add & norm again
        }
    }
    
    
    
}
