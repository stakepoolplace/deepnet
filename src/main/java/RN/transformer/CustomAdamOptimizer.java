package RN.transformer;

import java.io.Serializable;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;

public class CustomAdamOptimizer implements Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 3031098044411623634L;
	private transient Adam adamConfig;
    private transient AdamUpdater adamUpdater;
    private double initialLr;
    private int modelSize;
    private int warmupSteps;
    private int currentStep;
    private int epoch;
    private double learningRate;
    private INDArray stateViewArray; // Utilisé pour stocker l'état interne de l'optimiseur.

    // Constructeur avec initialisation de l'état de l'optimiseur.
    public CustomAdamOptimizer(double initialLr, int dmodel, int warmupSteps, long numberOfParameters) {
        this.initialLr = initialLr;
        this.modelSize = dmodel;
        this.learningRate = initialLr; // Initialisation explicite du taux d'apprentissage à la valeur initiale.
        this.warmupSteps = warmupSteps;
        this.currentStep = 0;
        this.adamConfig = new Adam(initialLr);
        
        // Initialisation de stateViewArray en fonction du nombre total de paramètres.
        this.stateViewArray = Nd4j.zeros(1, 2 * numberOfParameters); // 2 * car Adam utilise m et v.
        
        this.adamUpdater = new AdamUpdater(adamConfig);
        this.adamUpdater.setStateViewArray(this.stateViewArray, new long[]{numberOfParameters}, 'c', true);
    }


    
    public void update(List<INDArray> params, List<INDArray> grads) {
        if (params.size() != grads.size()) {
            throw new IllegalArgumentException("La taille de la liste des paramètres et des gradients doit être la même.");
        }
        
        for (int i = 0; i < params.size(); i++) {
            INDArray param = params.get(i);
            INDArray grad = grads.get(i); // Récupération du gradient correspondant au paramètre.
            
            calculateLearningRate(); // Calcul du taux d'apprentissage actuel.
            adamConfig.setLearningRate(this.learningRate);
            adamUpdater.setConfig(adamConfig);

            // L'ajustement et l'application du gradient spécifique au paramètre courant.
            adamUpdater.applyUpdater(grad, currentStep, epoch); // Notez que grad est utilisé directement ici.

            // Mise à jour du paramètre en soustrayant le gradient ajusté.
            param.subi(grad);
        }
        
        currentStep++; // Incrémentation du nombre de pas après la mise à jour.
    }


    // Calcul du taux d'apprentissage en fonction du pas actuel.
    public double calculateLearningRate() {
        double step = Math.max(1.0, (double) currentStep);  // Commencer à 1 pour éviter la division par zéro
        
        // Learning rate de base
        double lr = initialLr;
        
        // Calculer le facteur de warmup (0 à 1 pendant la période de warmup)
        double warmupFactor = Math.min(1.0, step / warmupSteps);
        
        // Calculer le facteur de décroissance (diminue après la période de warmup)
        double decayFactor;
        if (step <= warmupSteps) {
            decayFactor = 1.0;
        } else {
            // Décroissance en racine carrée inverse après le warmup
            decayFactor = Math.sqrt(warmupSteps / step);
        }
        
        // Calculer le learning rate final
        this.learningRate = lr * warmupFactor * decayFactor;
        
        // Appliquer des limites pour éviter des valeurs extrêmes
        this.learningRate = Math.max(this.learningRate, initialLr * 0.1);   // Pas moins de 10% du lr initial
        this.learningRate = Math.min(this.learningRate, initialLr);         // Pas plus que le lr initial
        
        // Log des valeurs pour debugging
        if (currentStep % 100 == 0) {
            System.out.printf("Step: %d, Warmup: %.3f, Decay: %.3f, LR: %.6f%n", 
                currentStep, warmupFactor, decayFactor, this.learningRate);
        }
        
        return this.learningRate;
    }


    // Getters et setters.
    public void setCurrentStep(int step) {
        this.currentStep = step;
    }

    public int getCurrentStep() {
        return this.currentStep;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public int getEpoch() {
        return this.epoch;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
