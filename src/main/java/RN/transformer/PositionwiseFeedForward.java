package RN.transformer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Classe représentant le réseau Feed-Forward positionnel dans le Transformer.
 * Utilise deux couches linéaires avec une activation ReLU entre elles.
 * Les tenseurs sont supposés avoir la forme [seqLength, dModel].
 */
public class PositionwiseFeedForward implements Serializable {
    private static final long serialVersionUID = 4036365276693483563L;
    private INDArray W1, b1, W2, b2;
    private INDArray inputCache, reluCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();
    private int dModel;

    /**
     * Constructeur de la classe PositionwiseFeedForward.
     * 
     * @param modelSize Taille du modèle (dModel)
     * @param ffSize    Taille de la couche Feed-Forward (ffSize)
     */
    public PositionwiseFeedForward(int modelSize, int ffSize) {
        this.dModel = modelSize;
        // Initialisation des poids avec une distribution uniforme ou normale selon le choix
        this.W1 = Nd4j.randn(modelSize, ffSize).div(Math.sqrt(modelSize)); // He Initialization
        this.b1 = Nd4j.zeros(1, ffSize);
        this.W2 = Nd4j.randn(ffSize, modelSize).div(Math.sqrt(ffSize)); // He Initialization
        this.b2 = Nd4j.zeros(1, modelSize);
    }

    /**
     * Passe forward du réseau Feed-Forward positionnel.
     * 
     * @param input Entrée de forme [batchSize, seqLength, dModel]
     * @return Sortie de forme [batchSize, seqLength, dModel]
     */
    public INDArray forward(INDArray input) {
        // Stockage de l'entrée pour la rétropropagation
        this.inputCache = input.dup();
        
        long batchSize = input.shape()[0];
        long seqLength = input.shape()[1];
        
        // Reshape input pour la multiplication matricielle
        INDArray inputReshaped = input.reshape(batchSize * seqLength, dModel);
        
        // Première couche linéaire
        INDArray hidden = inputReshaped.mmul(W1).addiRowVector(b1); // [(batchSize * seqLength), ffSize]
        this.reluCache = hidden.dup();
        
        // Activation ReLU
        INDArray reluOutput = Transforms.relu(hidden);
        
        // Deuxième couche linéaire
        INDArray output = reluOutput.mmul(W2).addiRowVector(b2); // [(batchSize * seqLength), dModel]
        
        // Reshape back to original dimensions
        return output.reshape(batchSize, seqLength, dModel);
    }

    /**
     * Passe backward pour calculer les gradients.
     * 
     * @param gradOutput Gradient provenant de la couche suivante de forme [seqLength, dModel]
     * @return Map contenant les gradients pour les paramètres 'W1', 'b1', 'W2', 'b2', et 'input'
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        // 1. Calcul des gradients par rapport à W2 et b2
        INDArray reluOutput = Transforms.relu(reluCache); // [seqLength, ffSize]
        INDArray gradW2 = reluOutput.transpose().mmul(gradOutput); // [ffSize, dModel]
        INDArray gradB2 = gradOutput.sum(0); // [dModel]

        // 2. Propagation du gradient à travers la deuxième couche linéaire
        INDArray gradHidden = gradOutput.mmul(W2.transpose()); // [seqLength, ffSize]

        // 3. Application de la dérivée de ReLU
        INDArray reluGrad = reluCache.gt(0).castTo(reluOutput.dataType()); // [seqLength, ffSize]
        INDArray gradThroughRelu = gradHidden.mul(reluGrad); // [seqLength, ffSize]

        // 4. Calcul des gradients par rapport à W1 et b1
        INDArray gradW1 = inputCache.transpose().mmul(gradThroughRelu); // [dModel, ffSize]
        INDArray gradB1 = gradThroughRelu.sum(0); // [ffSize]

        // 5. Calcul du gradient à propager à la couche précédente
        INDArray gradInput = gradThroughRelu.mmul(W1.transpose()); // [seqLength, dModel]

        // Stockage des gradients dans la map
        gradients.put("W1", gradW1);
        gradients.put("b1", gradB1);
        gradients.put("W2", gradW2);
        gradients.put("b2", gradB2);
        gradients.put("input", gradInput);

        return gradients;
    }

    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [W1, b1, W2, b2]
     */
    public List<INDArray> getGradients() {
        return Arrays.asList(gradients.get("W1"), gradients.get("b1"), gradients.get("W2"), gradients.get("b2"));
    }

    /**
     * Obtient les paramètres du réseau Feed-Forward positionnel.
     * 
     * @return Liste des paramètres dans l'ordre [W1, b1, W2, b2]
     */
    public List<INDArray> getParameters() {
        return Arrays.asList(W1, b1, W2, b2);
    }

    /**
     * Définit (met à jour) les paramètres du réseau Feed-Forward positionnel.
     * 
     * @param newW1 Nouvelles valeurs pour W1
     * @param newB1 Nouvelles valeurs pour b1
     * @param newW2 Nouvelles valeurs pour W2
     * @param newB2 Nouvelles valeurs pour b2
     */
    public void setParameters(INDArray newW1, INDArray newB1, INDArray newW2, INDArray newB2) {
        this.W1 = newW1;
        this.b1 = newB1;
        this.W2 = newW2;
        this.b2 = newB2;
    }

    /**
     * Obtient le nombre total de paramètres.
     * 
     * @return Nombre total de paramètres
     */
    public long getNumberOfParameters() {
        return W1.length() + b1.length() + W2.length() + b2.length();
    }

    /**
     * Obtient le nombre total de gradients.
     * 
     * @return Nombre total de gradients
     */
    public long getNumberOfGradients() {
        return gradients.get("W1").length() + gradients.get("b1").length() +
               gradients.get("W2").length() + gradients.get("b2").length();
    }
}
