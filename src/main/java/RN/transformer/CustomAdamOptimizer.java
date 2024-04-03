package RN.transformer;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    
    public CustomAdamOptimizer(double initialLr, int warmupSteps) {
        this.initialLr = initialLr;
        this.warmupSteps = warmupSteps;
        this.currentStep = 0;
        // Configuration d'Adam avec le taux d'apprentissage initial
        this.adamConfig = new Adam(initialLr);
        this.adamUpdater = new AdamUpdater(adamConfig);
    }

    public void update(List<INDArray> params, INDArray grads) {
        // Supposer que params et grads ont la même taille
        for (int i = 0; i < params.size(); i++) {
            INDArray param = params.get(i);

            // Calcule et ajuste le taux d'apprentissage
            this.learningRate = calculateLearningRate();
            adamConfig.setLearningRate(this.learningRate);
            adamUpdater.setConfig(adamConfig);

            // Création d'un faux gradient pour simuler la mise à jour des paramètres
            INDArray adjustedGrad = grads.dup(); // Dupliquer pour ajustement
            adamUpdater.applyUpdater(adjustedGrad, currentStep, epoch);

            // Mise à jour des paramètres en soustrayant le gradient ajusté
            param.subi(adjustedGrad);
        }

        currentStep++;
    }

    private double calculateLearningRate() {
        double step = currentStep + 1;
        double lrWarmup = initialLr * Math.min(1.0, step / warmupSteps);
        double lrDecay = initialLr * (Math.sqrt(warmupSteps) / Math.sqrt(step));

        // Choisissez le minimum entre lrWarmup et lrDecay pour avoir un échauffement suivi d'une décroissance
        return Math.min(lrWarmup, lrDecay);
    }



    // Getters et setters pour currentStep et epoch
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
