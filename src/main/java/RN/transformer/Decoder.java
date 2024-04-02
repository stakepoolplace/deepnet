package RN.transformer;

import java.util.ArrayList;
import java.util.Collection;
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
    private LinearProjection linearProjection; // Ajout de la projection linéaire


    public Decoder(int numLayers, int dModel, int numHeads, int dff, double dropoutRate, int outputSize) {
    	this.numLayers = numLayers;
    	this.dModel = dModel;
    	this.numHeads = numHeads;
    	this.dropoutRate = dropoutRate;
        this.layers = new ArrayList<>();
        this.layerNorm = new LayerNorm(dModel);
        this.linearProjection = new LinearProjection(dModel, outputSize); // Initialisation de la projection linéaire

        for (int i = 0; i < numLayers; i++) {
            this.layers.add(new DecoderLayer(dModel, numHeads, dff, dropoutRate));
        }
    }

    public INDArray forward(INDArray x, INDArray encoderOutput, INDArray lookAheadMask, INDArray paddingMask) {
        for (DecoderLayer layer : layers) {
            x = layer.forward(x, encoderOutput, lookAheadMask, paddingMask);
        }
        x = linearProjection.project(x); // Projection linéaire sur la sortie du décodeur

        return layerNorm.forward(x);
    }
    
    
    public List<List<Float>> decode(List<List<Float>> encodedInput) {
        List<List<Float>> decodedLogits = new ArrayList<>();

        // Pour chaque exemple encodé
        for (List<Float> encodedExample : encodedInput) {
            List<Float> exampleLogits = new ArrayList<>();

            // Initialiser le token de départ avec un token spécial de début de séquence
            List<Float> startToken = getStartToken();

            // Ajouter le token de départ aux logits de l'exemple
            exampleLogits.addAll(startToken);

            // Pour chaque token encodé dans l'exemple
            for (Float encodedToken : encodedExample) {
                // Appliquer l'attention sur le dernier token généré et les embeddings encodés
                List<Float> attentionWeights = calculateAttentionWeights(exampleLogits, encodedInput);

                // Calculer le contexte pondéré en utilisant les poids d'attention
                List<Float> contextVector = calculateContextVector(attentionWeights, encodedInput);

                // Ajouter les logits du prochain token aux logits de l'exemple
                exampleLogits.addAll(contextVector);
            }

            // Ajouter les logits de l'exemple à la liste des logits décodés
            decodedLogits.add(exampleLogits);
        }

        return decodedLogits;
    }


    // Méthodes auxiliaires pour le décodage

    private List<Float> getStartToken() {
        // Pour l'exemple, retourner un vecteur d'embeddings pour un token spécial de début de séquence
        List<Float> startToken = new ArrayList<>();
        for (int i = 0; i < dModel; i++) {
            startToken.add(0.0f); // Valeurs arbitraires pour l'exemple
        }
        return startToken;
    }

    private List<Float> calculateAttentionWeights(List<Float> previousLogits, List<List<Float>> encodedInput) {
        // Implémenter la logique pour calculer les poids d'attention en utilisant l'attention softmax
        // Pour cet exemple, nous supposerons que les poids d'attention sont simplement calculés comme une distribution softmax des logits du dernier token généré
        List<Float> attentionWeights = new ArrayList<>();
        float sumExpLogits = 0.0f;
        for (Float logit : previousLogits) {
            sumExpLogits += Math.exp(logit.floatValue());
        }
        for (Float logit : previousLogits) {
            attentionWeights.add((float) Math.exp(logit.floatValue()) / sumExpLogits);
        }
        return attentionWeights;
    }


    private List<Float> calculateContextVector(List<Float> attentionWeights, List<List<Float>> encodedInput) {
        // Pour cet exemple, le contexte pondéré est simplement la somme pondérée des embeddings encodés, où les poids d'attention servent de coefficients de pondération
        List<Float> contextVector = new ArrayList<>();
        int numDimensions = encodedInput.get(0).size(); // Supposons que tous les tokens ont la même dimension
        for (int i = 0; i < numDimensions; i++) {
            float sumWeightedEmbeddings = 0.0f;
            for (int j = 0; j < encodedInput.size(); j++) {
                sumWeightedEmbeddings += encodedInput.get(j).get(i) * attentionWeights.get(j);
            }
            contextVector.add(sumWeightedEmbeddings);
        }
        return contextVector;
    }

	 // Ajout à la classe Decoder
	
	 // Méthode pour obtenir tous les paramètres du décodeur
	 public List<INDArray> getParameters() {
		 
	        List<INDArray> params = new ArrayList<>();
	        
	        // Collecter les paramètres de chaque couche du décodeur
	        for (DecoderLayer layer : layers) {
	            params.addAll(layer.getParameters());
	        }

	        // Collecter les paramètres de la normalisation de couche finale
	        params.addAll(layerNorm.getParameters());

	        // Ajouter ici la collecte des paramètres d'autres composants, si nécessaire
	        
	        return params;
	 }
	
	 // Méthode pour calculer les gradients basés sur la perte
	 public INDArray calculateGradients(double loss) {
	     // Dans un cas réel, cette méthode impliquerait le calcul du gradient de la perte par rapport à chaque paramètre
	     // Pour cet exemple, simuler un gradient comme un INDArray de mêmes dimensions que les paramètres
	     INDArray gradients = Nd4j.rand(1, 100); // Assumer les mêmes dimensions hypothétiques que les paramètres
	     return gradients;
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

        public List<INDArray> getParameters() {
            List<INDArray> layerParams = new ArrayList<>();
            
            // Collecter les paramètres des mécanismes d'attention et du réseau feedforward
            layerParams.addAll(selfAttention.getParameters());
            layerParams.addAll(encoderDecoderAttention.getParameters());
            layerParams.addAll(feedForward.getParameters());

            // Collecter les paramètres des normalisations de couches
            layerParams.addAll(layerNorm1.getParameters());
            layerParams.addAll(layerNorm2.getParameters());
            layerParams.addAll(layerNorm3.getParameters());

            // Collecter les paramètres d'autres composants si nécessaire
            
            return layerParams;
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
    }
}
