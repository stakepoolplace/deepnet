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

import org.nd4j.linalg.api.buffer.DataType;

/**
 * Classe représentant une projection linéaire avec option de normalisation de couche (LayerNorm).
 * Les tenseurs sont supposés avoir la forme [batchSize, seqLength, dModel] ou [batchSize * seqLength, dModel].
 */
public class LinearProjection implements Serializable {
    
    private static final long serialVersionUID = -6601830517666118676L;
    private INDArray weights; // Poids de la projection linéaire [dModel, outputSize]
    private INDArray bias;    // Biais de la projection linéaire [1, outputSize]
    private INDArray gamma;   // Paramètre de scale pour LayerNorm [1, dModel]
    private INDArray beta;    // Paramètre de shift pour LayerNorm [1, dModel]
    private final double epsilon = 1e-7; // Petite constante pour éviter la division par zéro
    
    // Gradients calculés lors de la passe backward
    private Map<String, INDArray> gradients = null;
    private boolean useLayerNorm; // Contrôle de l'application de LayerNorm

    /**
     * Constructeur de la classe LinearProjection.
     * 
     * @param inputSize  Taille de l'entrée (dModel)
     * @param outputSize Taille de la sortie (par exemple, la taille du vocabulaire)
     * @param useLayerNorm Indique si LayerNorm doit être appliqué
     */
    public LinearProjection(int inputSize, int outputSize, boolean useLayerNorm) {
        this.useLayerNorm = useLayerNorm;
        // Initialisation des poids avec une distribution normale divisée par sqrt(inputSize) pour l'initialisation de He
        this.weights = Nd4j.randn(DataType.FLOAT, inputSize, outputSize).div(Math.sqrt(inputSize));
        // Initialisation des biais à zéro
        this.bias = Nd4j.zeros(DataType.FLOAT,1, outputSize);
        // Initialisation de gamma à un et beta à zéro pour la normalisation de couche
        this.gamma = Nd4j.ones(DataType.FLOAT,1, inputSize); // [1, dModel]
        this.beta = Nd4j.zeros(DataType.FLOAT,1, inputSize); // [1, dModel]
    }

    /**
     * Passe forward combinée pour la projection linéaire et la normalisation de couche.
     * 
     * @param input Entrée de forme [batchSize, seqLength, dModel] ou [batchSize * seqLength, dModel]
     * @return Sortie projetée de forme [batchSize, seqLength, outputSize] ou [batchSize * seqLength, outputSize]
     */
    public INDArray project(INDArray input) {
        
        INDArray scaled;
        
        // Vérifier si l'entrée est de rang 3
        boolean was3D = false;
        int batchSize = 1;
        int seqLength = 1;
        
        if (input.rank() == 3) {
            was3D = true;
            batchSize = (int) input.size(0);
            seqLength = (int) input.size(1);
            int inputDim = (int) input.size(2);
            
            // Reshaper en [batchSize * seqLength, inputDim]
            input = input.reshape(batchSize * seqLength, inputDim);
        } else if (input.rank() != 2) {
            throw new IllegalArgumentException("Input must be rank 2 or 3.");
        }
        
        // Application de LayerNorm si activé
        if (useLayerNorm) {
            // Calcul de la moyenne et de la variance sur la dimension dModel (axis=1)
            INDArray mean = input.mean(1).reshape(input.size(0), 1); // [batchSize * seqLength, 1]
            INDArray variance = input.var(false, 1).reshape(input.size(0), 1); // [batchSize * seqLength, 1]
            INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [batchSize * seqLength, 1]
            
            // Normalisation: (input - mean) / sqrt(var + epsilon)
            INDArray normalized = input.sub(mean).mul(stdInv); // [batchSize * seqLength, dModel]
            
            // Mise à l'échelle et décalage: normalized * gamma + beta
            scaled = normalized.mul(gamma).add(beta); // [batchSize * seqLength, dModel]
        } else {
            scaled = input;
        }
        
        // Projection linéaire
        INDArray output = scaled.mmul(weights).addRowVector(bias); // [batchSize * seqLength, outputSize]
        
        // Si l'entrée originale était de rang 3, reshaper de retour en [batchSize, seqLength, outputSize]
        if (was3D) {
            int totalSeqLength = (int) output.size(0); // batchSize * seqLength
            int outputSize = (int) output.size(1);
            int calculatedSeqLength = totalSeqLength / batchSize; // seqLength = 12 / 3 = 4
            
            // Vérifier que seqLength est un entier positif
            if (totalSeqLength % batchSize != 0) {
                throw new IllegalArgumentException("Impossible de reshaper output en [batchSize, seqLength, outputSize].");
            }
            
            output = output.reshape(batchSize, calculatedSeqLength, outputSize); // [3,4,5]
        }
        
        return output;
    }

    /**
     * Passe backward pour calculer les gradients.
     * 
     * @param input      Entrée originale utilisée dans la passe forward de forme [batchSize, seqLength, dModel] ou [batchSize * seqLength, dModel]
     * @param gradOutput Gradient provenant de la couche suivante de forme [batchSize, seqLength, outputSize] ou [batchSize * seqLength, outputSize]
     * @return Map contenant les gradients pour les paramètres 'weights', 'bias', 'gamma' et 'beta', ainsi que 'input'
     */
    public Map<String, INDArray> backward(INDArray input, INDArray gradOutput) {

        gradients = new HashMap<>();

        // Vérifier les formes
        if (input.rank() != 3 && input.rank() != 2) {
            throw new IllegalArgumentException("Input must be rank 2 or 3.");
        }
        if (input.size(input.rank() - 1) != weights.size(0)) { // dModel
            throw new IllegalArgumentException("Input size mismatch. Expected " + weights.size(0) + ", got " + input.size(input.rank() - 1));
        }
    
        // Si l'input est de rang 3, reshaper pour le traitement
        boolean reshaped = false;
        int batchSize = 1;
        int seqLength = 1;
        if (input.rank() == 3) {
            batchSize = (int) input.size(0);
            seqLength = (int) input.size(1);
            input = input.reshape(batchSize * seqLength, input.size(2)); // [batchSize * seqLength, dModel]
            gradOutput = gradOutput.reshape(batchSize * seqLength, gradOutput.size(2)); // [batchSize * seqLength, outputSize]
            reshaped = true;
        }
    
        // Projection linéaire
        // Gradients pour la projection linéaire
        INDArray gradScaled = gradOutput.mmul(weights.transpose()); // [batchSize * seqLength, dModel]
    
        // Gradients pour les biais
        INDArray gradBias = gradOutput.sum(0).reshape(1, gradOutput.size(1)); // [1, outputSize]
    
        // Gradients pour les poids
        INDArray gradWeights = input.transpose().mmul(gradOutput); // [dModel, outputSize]
    
        // Gradients pour LayerNorm
        INDArray gradGamma = null;
        INDArray gradBeta = null;
        INDArray gradNormalized = null;
        if (useLayerNorm) {
            // Calcul des moyennes et variances utilisées dans la passe forward
            INDArray mean = input.mean(1).reshape(input.size(0), 1); // [batchSize * seqLength, 1]
            INDArray variance = input.var(false, 1).reshape(input.size(0), 1); // [batchSize * seqLength, 1]
            INDArray stdInv = Transforms.pow(variance.add(epsilon), -0.5); // [batchSize * seqLength, 1]
        
            // Calcul de normalized = (input - mean) / sqrt(var + epsilon)
            INDArray normalized = input.sub(mean).mul(stdInv); // [batchSize * seqLength, dModel]
        
            // Gradients pour gamma et beta de LayerNorm
            gradGamma = normalized.mul(gradScaled).sum(0).reshape(1, input.size(1)); // [1, dModel]
            gradBeta = gradScaled.sum(0).reshape(1, input.size(1)); // [1, dModel]
        
            // Gradients pour la normalisation
            gradNormalized = gradScaled.mul(gamma); // [batchSize * seqLength, dModel]
        
            // Calcul des gradients pour l'entrée
            INDArray sumGradNormInput = gradNormalized.mul(normalized).sum(1).reshape(input.size(0), 1); // [batchSize * seqLength, 1]
            INDArray gradInput = gradNormalized.mul(stdInv).sub(normalized.mul(sumGradNormInput).mul(stdInv)); // [batchSize * seqLength, dModel]
        
            // Remettre les formes d'origine si nécessaire
            if (reshaped) {
                gradInput = gradInput.reshape(batchSize, seqLength, input.size(1)); // [batchSize, seqLength, dModel]
            }
        
            // Stockage des gradients dans la map
            NDArrayUtils.addGradient(gradients, "weights", gradWeights);
            NDArrayUtils.addGradient(gradients, "bias", gradBias);
            NDArrayUtils.addGradient(gradients, "gamma", gradGamma);
            NDArrayUtils.addGradient(gradients, "beta", gradBeta);
            NDArrayUtils.addGradient(gradients, "input", gradInput); // Gradient à propager vers les couches précédentes
        } else {
            // Si LayerNorm n'est pas utilisé, les gradients sont simplement ceux de la projection linéaire
            INDArray gradInput = gradScaled;
            // Remettre les formes d'origine si nécessaire
            if (reshaped) {
                gradInput = gradInput.reshape(batchSize, seqLength, input.size(1)); // [batchSize, seqLength, dModel]
            }
            // Stockage des gradients dans la map
            NDArrayUtils.addGradient(gradients, "weights", gradWeights);
            NDArrayUtils.addGradient(gradients, "bias", gradBias);
            NDArrayUtils.addGradient(gradients, "input", gradInput); // Gradient à propager vers les couches précédentes
        }
    
        return gradients;
    }
    
    
    /**
     * Obtient les gradients des paramètres.
     * 
     * @return Liste des gradients dans l'ordre [weights, bias, gamma, beta]
     */
    public List<INDArray> getGradients() {
        List<INDArray> list = new ArrayList<>();
        list.add(gradients.get("weights"));
        list.add(gradients.get("bias"));
        if (useLayerNorm) {
            list.add(gradients.get("gamma"));
            list.add(gradients.get("beta"));
        }
        if (list.contains(null)) {
            throw new IllegalArgumentException("Gradients contiennent des valeurs nulles.");
        }
        return list;    
    }

    /**
     * Obtient les paramètres de la projection linéaire.
     * 
     * @return Liste des paramètres dans l'ordre [weights, bias, gamma, beta]
     */
    public List<INDArray> getParameters() {
        List<INDArray> params = new ArrayList<>();
        params.add(weights);
        params.add(bias);
        if (useLayerNorm) {
            params.add(gamma);
            params.add(beta);
        }
        return params;
    }

    /**
     * Définit (met à jour) les paramètres de la projection linéaire.
     * 
     * @param newWeights Nouvelles valeurs pour les poids
     * @param newBias    Nouvelles valeurs pour les biais
     * @param newGamma   Nouvelles valeurs pour gamma (si LayerNorm est utilisé)
     * @param newBeta    Nouvelles valeurs pour beta (si LayerNorm est utilisé)
     */
    public void setParameters(INDArray newWeights, INDArray newBias, INDArray newGamma, INDArray newBeta) {
        setWeights(newWeights, newBias);
        if (useLayerNorm) {
            if (newGamma != null && newBeta != null) {
                this.gamma = newGamma.dup();
                this.beta = newBeta.dup();
            } else {
                throw new IllegalArgumentException("Gamma et Beta ne peuvent pas être null si LayerNorm est utilisé.");
            }
        }
    }

    /**
     * Obtient le nombre total de paramètres.
     * 
     * @return Nombre total de paramètres
     */
    public long getNumberOfParameters() {
        long count = weights.length() + bias.length();
        if (useLayerNorm) {
            count += gamma.length() + beta.length();
        }
        return count;
    }

    /**
     * Obtient le nombre total de gradients.
     * 
     * @return Nombre total de gradients
     */
    public long getNumberOfGradients() {
        long count = gradients.get("weights").length() + gradients.get("bias").length();
        if (useLayerNorm) {
            count += gradients.get("gamma").length() + gradients.get("beta").length();
        }
        return count;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    /** 
     * Définit (met à jour) les poids et biais de la projection linéaire.
     * 
     * @param newWeights Nouvelles valeurs pour les poids [dModel, outputSize]
     * @param newBias    Nouvelles valeurs pour les biais [1, outputSize]
     */
    public void setWeights(INDArray newWeights, INDArray newBias) {
        // Vérifier les dimensions des nouveaux poids
        if (!Arrays.equals(newWeights.shape(), this.weights.shape())) {
            throw new IllegalArgumentException("La forme des nouveaux poids ne correspond pas à la forme existante.");
        }
        
        // Vérifier les dimensions des nouveaux biais
        if (!Arrays.equals(newBias.shape(), this.bias.shape())) {
            throw new IllegalArgumentException("La forme des nouveaux biais ne correspond pas à la forme existante.");
        }
        
        this.weights = newWeights.dup();
        this.bias = newBias.dup();
        
    }

    public INDArray getWeights() {
        return weights;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
    }

    public INDArray getBias() {
        return bias;
    }

    public void setGamma(INDArray gamma) {
        if (useLayerNorm) {
            this.gamma = gamma;
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Impossible de définir gamma.");
        }
    }

    public INDArray getGamma() {
        if (useLayerNorm) {
            return gamma;
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Impossible d'obtenir gamma.");
        }
    }

    public void setBeta(INDArray beta) {
        if (useLayerNorm) {
            this.beta = beta;
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Impossible de définir beta.");
        }
    }

    public INDArray getBeta() {
        if (useLayerNorm) {
            return beta;
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Impossible d'obtenir beta.");
        }
    }

    /**
     * Obtient le gradient de la perte par rapport aux poids.
     * 
     * @return Gradient par rapport aux poids [dModel, outputSize]
     */
    public INDArray getdLoss_dWeights() {
        if (gradients.containsKey("weights")) {
            return gradients.get("weights");
        } else {
            throw new IllegalStateException("Le gradient des poids n'a pas été calculé. Appelez backward() d'abord.");
        }
    }

    /**
     * Obtient le gradient de la perte par rapport aux biais.
     * 
     * @return Gradient par rapport aux biais [1, outputSize]
     */
    public INDArray getdLoss_dBias() {
        if (gradients.containsKey("bias")) {
            return gradients.get("bias");
        } else {
            throw new IllegalStateException("Le gradient des biais n'a pas été calculé. Appelez backward() d'abord.");
        }
    }

    /**
     * Obtient le gradient de la perte par rapport à gamma (si LayerNorm est utilisé).
     * 
     * @return Gradient par rapport à gamma [1, dModel]
     */
    public INDArray getdLoss_dGamma() {
        if (useLayerNorm) {
            if (gradients.containsKey("gamma")) {
                return gradients.get("gamma");
            } else {
                throw new IllegalStateException("Le gradient de gamma n'a pas été calculé. Appelez backward() d'abord.");
            }
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Aucun gradient pour gamma.");
        }
    }

    /**
     * Obtient le gradient de la perte par rapport à beta (si LayerNorm est utilisé).
     * 
     * @return Gradient par rapport à beta [1, dModel]
     */
    public INDArray getdLoss_dBeta() {
        if (useLayerNorm) {
            if (gradients.containsKey("beta")) {
                return gradients.get("beta");
            } else {
                throw new IllegalStateException("Le gradient de beta n'a pas été calculé. Appelez backward() d'abord.");
            }
        } else {
            throw new UnsupportedOperationException("LayerNorm n'est pas activé. Aucun gradient pour beta.");
        }
    }


}