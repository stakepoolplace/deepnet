package RN.transformer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class CustomAdamOptimizer implements Serializable {
    private static final long serialVersionUID = 3031098044411623634L;

    // Hyperparamètres d'Adam
    private double initialLr;
    private double beta1;
    private double beta2;
    private double epsilon;

    // Paramètres de scheduling
    private int warmupSteps;
    private int currentStep;
    private int epoch;
    private double learningRate;

    // États pour chaque paramètre
    private List<AdamState> states;

    // Classe interne pour stocker m et v
    private static class AdamState implements Serializable {
        private static final long serialVersionUID = 1L;
        public INDArray m;
        public INDArray v;

        public AdamState(long[] shape) {
            this.m = Nd4j.zeros(DataType.FLOAT, shape);
            this.v = Nd4j.zeros(DataType.FLOAT, shape);
        }
    }

    /**
     * Constructeur de l'optimiseur Adam.
     *
     * @param initialLr        Taux d'apprentissage initial.
     * @param dmodel           Dimension du modèle (dModel).
     * @param warmupSteps      Nombre de pas de warmup.
     * @param params           Liste des paramètres du modèle.
     */
    public CustomAdamOptimizer(double initialLr, int dmodel, int warmupSteps, List<INDArray> params) {
        this.initialLr = initialLr;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.warmupSteps = warmupSteps;
        this.currentStep = 0;
        this.epoch = 0;
        this.learningRate = initialLr;

        // Initialiser les états Adam pour chaque paramètre
        this.states = new ArrayList<>();
        for (INDArray param : params) {
            this.states.add(new AdamState(param.shape()));
        }
    }

    // Classe interne pour gérer la sérialisation des états Adam
    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject();
        // Sérialiser les états Adam
        oos.writeInt(states.size());
        for (AdamState state : states) {
            oos.writeObject(state.m);
            oos.writeObject(state.v);
        }
    }

    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        ois.defaultReadObject();
        // Désérialiser les états Adam
        int size = ois.readInt();
        this.states = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            INDArray m = (INDArray) ois.readObject();
            INDArray v = (INDArray) ois.readObject();
            AdamState state = new AdamState(m.shape());
            state.m = m;
            state.v = v;
            this.states.add(state);
        }
    }

    // Méthode pour mettre à jour les paramètres
    public void update(List<INDArray> params, List<INDArray> grads) {
        if (params.size() != grads.size()) {
            throw new IllegalArgumentException("La taille de la liste des paramètres et des gradients doit être la même.");
        }

        currentStep++;

        for (int i = 0; i < params.size(); i++) {
            INDArray param = params.get(i);
            INDArray grad = grads.get(i);

            AdamState state = states.get(i);

            // Mise à jour des moments
            state.m.mul(beta1).addi(grad.mul(1 - beta1));
            state.v.mul(beta2).addi(grad.mul(grad).mul(1 - beta2));

            // Correction de biais
            INDArray mHat = state.m.mul(1.0 / (1 - Math.pow(beta1, currentStep)));
            INDArray vHat = state.v.mul(1.0 / (1 - Math.pow(beta2, currentStep)));

            // Calcul de l'étape de mise à jour
            INDArray step = mHat.mul(learningRate).div(Transforms.sqrt(vHat).add(epsilon));

            // Mise à jour du paramètre
            param.subi(step);
        }

        // Mettre à jour le taux d'apprentissage si nécessaire
        calculateLearningRate();
    }

    // Calcul du taux d'apprentissage avec warmup et décroissance
    double calculateLearningRate() {
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

    // Getters et setters
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
