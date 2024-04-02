package RN.transformer;

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

	    public CustomAdamOptimizer(double initialLr, int warmupSteps) {
	        this.initialLr = initialLr;
	        this.warmupSteps = warmupSteps;
	        this.currentStep = 0;
	        // Configuration d'Adam avec le taux d'apprentissage initial
	        this.adamConfig = new Adam(initialLr);
	        this.adamUpdater = new AdamUpdater(adamConfig);
	    }

	    public void update(INDArray params, INDArray grads, float loss) {
	        // Calcule le taux d'apprentissage ajusté
	        double adjustedLr = calculateLearningRate(loss);
	        adamConfig.setLearningRate(adjustedLr);
	        
	        // Ajuster les gradients en fonction du taux d'apprentissage ajusté
	        INDArray update = grads.mul(adjustedLr); // Simplification: applique le lr directement aux gradients
	        
	        // Mettre à jour les paramètres
	        params.subi(update);
	        
	        // Mettre à jour les paramètres de l'optimizer
	        adamUpdater.applyUpdater(grads, this.currentStep, this.epoch);
	        currentStep++;
	    }

    private double calculateLearningRate(float loss) {
        double step = currentStep + 1;
        double lrWarmup = initialLr * Math.min(1.0, step / warmupSteps);
        double lrDecay = initialLr * Math.sqrt(Math.max(step, warmupSteps) / warmupSteps);

        // Adapter le taux d'apprentissage en fonction de la valeur de la perte
        // Par exemple, on peut réduire le taux d'apprentissage si la perte diminue
        double adjustedLr = Math.min(lrWarmup, lrDecay);
        if (loss < 0.1) {
            adjustedLr *= 0.1; // Réduction du taux d'apprentissage si la perte est inférieure à 0.1
        }

        return adjustedLr;
    }

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
}
