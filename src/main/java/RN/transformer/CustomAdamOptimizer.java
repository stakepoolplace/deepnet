package RN.transformer;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;

public class CustomAdamOptimizer {
    private Adam adamConfig;
    private AdamUpdater adamUpdater;
    private double initialLr;
    private int warmupSteps;
    private int currentStep;
    private int epoch;
    private double learningRate;
    private INDArray stateViewArray; // Utilisé pour stocker l'état interne de l'optimiseur.

    // Constructeur avec initialisation de l'état de l'optimiseur.
    public CustomAdamOptimizer(double initialLr, int warmupSteps, long numberOfParameters) {
        this.initialLr = initialLr;
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
            
            this.learningRate = calculateLearningRate(); // Calcul du taux d'apprentissage actuel.
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
        double step = currentStep + 1;
        double lrWarmup = initialLr * Math.min(1.0, step / warmupSteps); // Augmentation pendant le warmup.
        double lrDecay = initialLr * (Math.sqrt(warmupSteps) / Math.sqrt(step)); // Décroissance après warmup.

        return Math.min(lrWarmup, lrDecay);
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
