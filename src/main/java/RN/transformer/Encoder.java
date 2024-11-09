package RN.transformer;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


/**
 * 	numLayers: Le nombre de couches répétitives dans l'encodeur.
 *	dModel: La dimensionnalité des embeddings de tokens et des sorties de toutes les couches dans le modèle.
 *	numHeads: Le nombre de têtes d'attention dans les mécanismes d'attention multi-têtes.
 *	dff: La dimensionnalité des couches feed-forward internes dans chaque couche d'encodeur.
 *	vocabSize: La taille du vocabulaire, nécessaire pour les embeddings de tokens.
 *	maxSeqLength: La longueur maximale de séquence, utilisée pour les embeddings positionnels.
 */

public class Encoder implements Serializable  {
	
    /**
	 * 
	 */
	private static final long serialVersionUID = -5716799542280937448L;
	private List<EncoderLayer> layers;
    private int dModel;
    private PositionalEncoding positionalEncoding;
    private LayerNorm layerNorm;
    private Tokenizer tokenizer;

    public Encoder() {
	}
    
    public Encoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, Tokenizer tokenizer) {
    	this.dModel = dModel;
        this.positionalEncoding = new PositionalEncoding(dModel);
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.tokenizer = tokenizer;
        
        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new EncoderLayer(dModel, numHeads, dff, dropoutRate));
        }
    }



	public List<List<Float>> encode(boolean isTraining, String text) {
        // Tokenization du texte
        List<String> tokens = tokenizer.tokenize(text);
        // Conversion des tokens en IDs
        List<Integer> tokenIds = tokenizer.tokensToIds(tokens);

        // Encodage des IDs de tokens à travers les couches de l'encodeur
        INDArray inputEmbeddings = lookupEmbeddings(Arrays.asList(tokenIds));
        INDArray encoded = forward(isTraining, inputEmbeddings, null);
        
        // Conversion des embeddings encodés en logits
        return convertToLogits(encoded);
    }
    
    public INDArray encode(boolean isTraining, List<List<Integer>> tokenIdsBatch, INDArray paddingMask) {
        int batchSize = tokenIdsBatch.size();
        int seqLength = tokenIdsBatch.get(0).size();
        INDArray inputEmbeddings = Nd4j.zeros(DataType.FLOAT, batchSize, seqLength, dModel);

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                int tokenId = tokenIdsBatch.get(i).get(j);
                INDArray tokenEmbedding = TransformerModel.getPretrainedEmbeddings().getRow(tokenId);
                // Utilisez slice pour accéder à la position correcte et assign pour copier les valeurs
                inputEmbeddings.get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all())
                            .assign(tokenEmbedding);
            }
        }
        // System.out.println("Embedded input shape: " + Arrays.toString(inputEmbeddings.shape()));

        // Appliquer le MultiHeadAttention et les couches de l'encodeur
        INDArray encoded = forward(isTraining, inputEmbeddings, paddingMask);
        return encoded;
    }
    
    
    public INDArray lookupEmbeddings(List<List<Integer>> tokenIdsBatch) {
        int batchSize = tokenIdsBatch.size();
        int seqLength = tokenIdsBatch.get(0).size();
    
        // Étape 1: Aplatir la liste de listes en un tableau 1D d'entiers (int[])
        int[] flattenedTokenIds = new int[batchSize * seqLength];
        for (int i = 0; i < batchSize; i++) {
            List<Integer> tokenIds = tokenIdsBatch.get(i);
            for (int j = 0; j < seqLength; j++) {
                flattenedTokenIds[i * seqLength + j] = tokenIds.get(j); // Pas besoin de convertir en long
            }
        }
    
        // Étape 2: Récupérer les embeddings correspondants
        // TransformerModel.getPretrainedEmbeddings() doit être de forme [vocabSize, dModel]
        INDArray batchEmbeddings = TransformerModel.getPretrainedEmbeddings().getRows(flattenedTokenIds); // [batchSize * seqLength, dModel]
    
        // Étape 3: Reshaper en [batchSize, seqLength, dModel]
        batchEmbeddings = batchEmbeddings.reshape(batchSize, seqLength, dModel); // Utilisation correcte de dModel
    
        return batchEmbeddings;
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

    public INDArray forward(boolean isTraining, INDArray x, INDArray paddingMask) {
        
        // Appliquer les embeddings positionnels
        INDArray posEncoding = positionalEncoding.getPositionalEncoding(x.shape()[1]);
        x = x.add(posEncoding);
        // System.out.println("After positional encoding: " + Arrays.toString(x.shape()));
    
        for (EncoderLayer layer : layers) {
            x = layer.forward(isTraining, x, paddingMask);
            // System.out.println("After encoder layer: " + Arrays.toString(x.shape()));
        }
        
        x = layerNorm.forward(x);
        // System.out.println("After layer normalization: " + Arrays.toString(x.shape()));
    
        return x;
    }
    
    

    public void backward(Map<String, INDArray> gradOutput) {
    	
        // Récupérer gradAttentionOutputConcat
        INDArray gradAttentionOutputConcatND = gradOutput.get("gradAttentionOutputConcat"); // [1,6,6,50]

        // Permute les axes pour [batchSize, seqLength, numHeads, depth]
        INDArray gradPermuted = gradAttentionOutputConcatND.permute(0, 2, 1, 3); // [1,6,6,50]
    
        // Reshaper pour concaténer les têtes : [batchSize, seqLength, numHeads * depth]
        INDArray gradOutputForLayerNorm = gradPermuted.reshape(1, 6, 6 * 50); // [1,6,300]
    
        // Backpropagation à travers la normalisation de couche finale
    	Map<String, INDArray> gradientsFromLayerNorm = layerNorm.backward(gradOutputForLayerNorm);

        // Récupération du gradient par rapport aux entrées de LayerNorm qui sera utilisé comme gradient initial pour les couches de l'Encoder
        INDArray gradInput = gradientsFromLayerNorm.get("input");

        // Propagation des gradients à travers chaque couche d'Encoder en ordre inverse
        for (int i = layers.size() - 1; i >= 0; i--) {
            // Chaque couche retourne le gradient par rapport à ses entrées qui est passé à la couche précédente
            gradInput = layers.get(i).backward(gradInput);
            if (gradInput == null) {
                throw new IllegalArgumentException("gradInput est null après backward de la couche " + i);
            }
        }

        // Mettre à jour ou enregistrer les gradients pour gamma et beta si nécessaire
        // Par exemple, si ces paramètres sont appris :
        // updateGammaBeta(gradFromLayerNorm.get("gamma"), gradFromLayerNorm.get("beta"));
    }

    // Méthode hypothétique pour mettre à jour ou enregistrer les gradients de gamma et beta
    private void updateGammaBeta(INDArray gradGamma, INDArray gradBeta) {
        // Mettre à jour ou enregistrer les gradients de gamma et beta
        // Ceci pourrait inclure l'application d'un taux d'apprentissage ou l'enregistrement pour une utilisation dans un pas d'optimisation
    }




    
    // Méthode pour calculer les gradients basés sur la perte
    public INDArray calculateGradients(double loss) {
        // Dans un cas réel, cette méthode impliquerait le calcul du gradient de la perte par rapport à chaque paramètre
        // Pour cet exemple, simuler un gradient comme un INDArray de mêmes dimensions que les paramètres
        INDArray gradients = Nd4j.rand(1, 100); // Assumer les mêmes dimensions hypothétiques que les paramètres
        return gradients;
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
    
    public List<INDArray> getGradients() {
        List<INDArray> grads = new ArrayList<>();
        // Collecter les poids et biais de multiHeadAttention et positionwiseFeedForward
        for (EncoderLayer layer : layers) {
        	grads.addAll(layer.getGradients());
        }

        // Inclure les gradients de la normalisation de couche finale
        if(layerNorm != null) {
        	grads.addAll(layerNorm.getGradients());
        }
        
        return grads;
    }
    

    
    public int getNumberOfParameters() {
        int numParams = 0;

        // Parcourir toutes les couches d'encodeur pour compter leurs paramètres
        for (EncoderLayer layer : layers) {
            numParams += layer.getNumberOfParameters();
        }

        // Ajouter les paramètres de la normalisation de couche et des embeddings positionnels
        numParams += layerNorm.getNumberOfParameters();

        return numParams;
    }
    
    
    public int getNumberOfGradients() {
        int numGrads = 0;

        // Parcourir toutes les couches d'encodeur pour compter leurs gradients
        for (EncoderLayer layer : layers) {
        	numGrads += layer.getNumberOfGradients();
        }

        // Ajouter les gradients de la normalisation de couche et des embeddings positionnels
        numGrads += layerNorm.getNumberOfGradients();

        return numGrads;
    }



    static class EncoderLayer implements Serializable {
    	
        /**
		 * 
		 */
		private static final long serialVersionUID = -88886021425567141L;
		
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
        
        public INDArray forward(boolean isTraining, INDArray x, INDArray paddingMask) {
        	
            INDArray attnOutput = selfAttention.forward(x, x, x, paddingMask);
            attnOutput = dropout1.apply(isTraining, attnOutput);
            x = layerNorm1.forward(x.add(attnOutput)); // Add & norm

            INDArray ffOutput = feedForward.forward(x);
            ffOutput = dropout2.apply(isTraining, ffOutput);
            return layerNorm2.forward(x.add(ffOutput)); // Add & norm again
        }
        
        public INDArray backward(INDArray gradOutput) {
        	
            // Backward à travers la deuxième normalisation de couche
            Map<String, INDArray> gradLayerNorm2 = layerNorm2.backward(gradOutput);
            INDArray gradToFeedForward = gradLayerNorm2.get("input");

            // Backward à travers la couche PositionwiseFeedForward
            Map<String, INDArray> gradFeedForward = feedForward.backward(gradToFeedForward);
            INDArray gradToLayerNorm1 = gradFeedForward.get("input");

            // Backward à travers la première normalisation de couche
            Map<String, INDArray> gradLayerNorm1 = layerNorm1.backward(gradToLayerNorm1);
            INDArray gradToSelfAttention = gradLayerNorm1.get("input");

            // Backward à travers SelfAttention
            Map<String, INDArray> gradSelfAttention = selfAttention.backward(gradToSelfAttention);


            return gradSelfAttention.get("input");
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
        
        public List<INDArray> getGradients() {
            List<INDArray> layerGrads = new ArrayList<>();
            
            // Collecter les paramètres des composants de la couche d'encodeur
            layerGrads.addAll(selfAttention.getGradients());
            layerGrads.addAll(feedForward.getGradients());
            layerGrads.addAll(layerNorm1.getGradients());
            layerGrads.addAll(layerNorm2.getGradients());

            return layerGrads;
        }
        

        public long getNumberOfParameters() {
            return selfAttention.getNumberOfParameters() +
                   feedForward.getNumberOfParameters() +
                   layerNorm1.getNumberOfParameters() +
                   layerNorm2.getNumberOfParameters();
        }
        
        

        public long getNumberOfGradients() {
            return selfAttention.getNumberOfGradients() +
                   feedForward.getNumberOfGradients() +
                   layerNorm1.getNumberOfGradients() +
                   layerNorm2.getNumberOfGradients();
        }



    }
    
    
    
}
