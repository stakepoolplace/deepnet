package RN.transformer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import RN.utils.NDArrayUtils;

/**
 * Classe représentant le réseau Feed-Forward positionnel dans le Transformer.
 * Utilise deux couches linéaires avec une activation ReLU entre elles.
 * Les tenseurs sont supposés avoir la forme [batchSize, seqLength, dModel].
 */
public class PositionwiseFeedForward implements Serializable {
    private static final long serialVersionUID = 4036365276693483563L;
    private INDArray W1, b1, W2, b2;
    private INDArray inputCache, reluCache; // Cache pour le forward
    private Map<String, INDArray> gradients = new HashMap<>();
    private int dModel;
    private int ffSize; // Taille de la couche Feed-Forward
    private double epsilon = 1e-7; // Ajouter epsilon ici pour éviter la division par zéro

    /**
     * Constructeur de la classe PositionwiseFeedForward.
     * 
     * @param modelSize Taille du modèle (dModel)
     * @param ffSize    Taille de la couche Feed-Forward (ffSize)
     */
    public PositionwiseFeedForward(int modelSize, int ffSize) {
        this.dModel = modelSize;
        this.ffSize = ffSize;
        // He Initialization pour W1
        this.W1 = Nd4j.randn(modelSize, ffSize).mul(Math.sqrt(2.0 / modelSize)); // [dModel, ffSize]
        this.b1 = Nd4j.zeros(1, ffSize); // [1, ffSize]
        // Xavier Initialization pour W2
        this.W2 = Nd4j.randn(ffSize, modelSize).mul(Math.sqrt(1.0 / ffSize)); // [ffSize, dModel]
        this.b2 = Nd4j.zeros(1, modelSize); // [1, dModel]
    }
    

    /**
     * Passe forward du réseau Feed-Forward positionnel.
     * 
     * @param input Entrée de forme [batchSize, seqLength, dModel]
     * @return Sortie de forme [batchSize, seqLength, dModel]
     */
    public INDArray forward(INDArray input) {
        // Stockage de l'entrée pour la rétropropagation
        this.inputCache = input.dup(); // [batchSize, seqLength, dModel]
        
        long batchSize = input.shape()[0];
        long seqLength = input.shape()[1];
        
        // Reshape input pour la multiplication matricielle
        INDArray inputReshaped = input.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
        
        // Première couche linéaire
        INDArray hidden = inputReshaped.mmul(W1).addiRowVector(b1); // [batchSize * seqLength, ffSize]
        this.reluCache = hidden.dup(); // [batchSize * seqLength, ffSize]
        
        // Activation ReLU
        INDArray reluOutput = Transforms.relu(hidden); // [batchSize * seqLength, ffSize]
        
        // Deuxième couche linéaire
        INDArray output = reluOutput.mmul(W2).addiRowVector(b2); // [batchSize * seqLength, dModel]
        
        // Reshape back to original dimensions
        return output.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
    }

    /**
     * Passe backward pour calculer les gradients.
     * 
     * @param gradOutput Gradient provenant de la couche suivante de forme [batchSize, seqLength, dModel] ou [batchSize * seqLength, dModel]
     * @return Map contenant les gradients pour les paramètres 'W1', 'b1', 'W2', 'b2', et 'input'
     */
    public Map<String, INDArray> backward(INDArray gradOutput) {
        boolean reshaped = false;
        long batchSize = 1;
        long seqLength = 1;
        
        // Vérifier le rang de gradOutput et reshaper si nécessaire
        if (gradOutput.rank() == 3) {
            batchSize = gradOutput.size(0);
            seqLength = gradOutput.size(1);
            gradOutput = gradOutput.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]
            reshaped = true;
        } else if (gradOutput.rank() != 2) {
            throw new IllegalArgumentException("gradOutput doit être de rang 2 ou 3.");
        }

        // Reshape inputCache de la même manière que gradOutput
        INDArray inputReshaped = inputCache.reshape(batchSize * seqLength, dModel); // [batchSize * seqLength, dModel]

        // 1. Calcul des gradients par rapport à W2 et b2
        INDArray reluOutput = Transforms.relu(reluCache); // [batchSize * seqLength, ffSize]
        INDArray gradW2 = reluOutput.transpose().mmul(gradOutput); // [ffSize, dModel]
        INDArray gradB2 = gradOutput.sum(0).reshape(1, dModel); // [1, dModel]

        // 2. Propagation du gradient à travers la deuxième couche linéaire
        INDArray gradHidden = gradOutput.mmul(W2.transpose()); // [batchSize * seqLength, ffSize]

        // 3. Application de la dérivée de ReLU
        INDArray reluGrad = reluCache.gt(0).castTo(reluOutput.dataType()); // [batchSize * seqLength, ffSize]
        INDArray gradThroughRelu = gradHidden.mul(reluGrad); // [batchSize * seqLength, ffSize]

        // 4. Calcul des gradients par rapport à W1 et b1
        INDArray gradW1 = inputReshaped.transpose().mmul(gradThroughRelu); // [dModel, ffSize]
        INDArray gradB1 = gradThroughRelu.sum(0).reshape(1, ffSize); // [1, ffSize]
        
        // 5. Calcul du gradient à propager à la couche précédente
        INDArray gradInput = gradThroughRelu.mmul(W1.transpose()); // [batchSize * seqLength, dModel]

        // Remettre la forme d'origine si reshaped
        if (reshaped) {
            gradInput = gradInput.reshape(batchSize, seqLength, dModel); // [batchSize, seqLength, dModel]
        }

        // Stockage des gradients dans la map
        NDArrayUtils.addGradient(gradients, "W1", gradW1);
        NDArrayUtils.addGradient(gradients, "b1", gradB1);
        NDArrayUtils.addGradient(gradients, "W2", gradW2);
        NDArrayUtils.addGradient(gradients, "b2", gradB2);
        NDArrayUtils.addGradient(gradients, "input", gradInput); // Gradient à propager vers les couches précédentes

        return gradients;
    }

    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [W1, b1, W2, b2]
     */
    public List<INDArray> getGradients() {
        List<INDArray> list = new ArrayList<>();
        list.add(gradients.get("W1"));
        list.add(gradients.get("b1"));
        list.add(gradients.get("W2"));
        list.add(gradients.get("b2"));
        if (list.contains(null)) {
            throw new IllegalArgumentException(" gradients contains null ");
        }
        return list; 
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
