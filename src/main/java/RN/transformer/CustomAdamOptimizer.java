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
    private float initialLr;
    private float beta1;
    private float beta2;
    private float epsilon;

    // Paramètres de scheduling
    private int warmupSteps;
    private int currentStep;
    private int epoch;
    private float learningRate;

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
    public CustomAdamOptimizer(float initialLr, int dmodel, int warmupSteps, List<INDArray> params) {
        this.initialLr = initialLr;
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.epsilon = 1e-8f;
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
    
        if (params.size() != states.size()) {
            throw new IllegalStateException("Le nombre de paramètres (" + params.size() + ") ne correspond pas au nombre d'états (" + states.size() + "). Assurez-vous que l'optimiseur est initialisé après avoir ajouté tous les paramètres.");
        }
    
        currentStep++;
        // System.out.println("Current Step: " + currentStep);
        // System.out.println("Learning Rate: " + learningRate);
    
        for (int i = 0; i < params.size(); i++) {
            INDArray param = params.get(i);
            INDArray grad = grads.get(i);
        
            AdamState state = states.get(i);
        
            // Vérifier l'alignement
            // if (param != combinedParameters.get(i)) {
            //     System.err.println("Mauvais alignement des paramètres à l'index " + i);
            // }
        
            // Mise à jour des moments en place
            state.m.muli(beta1).addi(grad.mul(1 - beta1));
            state.v.muli(beta2).addi(grad.mul(grad).mul(1 - beta2));
        
            // Correction de biais
            INDArray mHat = state.m.mul(1.0f / (1.0f - (float) Math.pow(beta1, currentStep)));
            INDArray vHat = state.v.mul(1.0f / (1.0f - (float) Math.pow(beta2, currentStep)));
        
            // Calcul de l'étape de mise à jour
            INDArray step = mHat.mul(learningRate).div(Transforms.sqrt(vHat).add(epsilon));
        
            // Log des valeurs intermédiaires
            // System.out.println("Paramètre " + i + " avant mise à jour: " + param);
            // System.out.println("Gradient " + i + ": " + grad);
            // System.out.println("mHat " + i + ": " + mHat);
            // System.out.println("vHat " + i + ": " + vHat);
            // System.out.println("Step " + i + ": " + step);
        
            // Mise à jour du paramètre
            param.subi(step);
        
            // Log après mise à jour
            // System.out.println("Paramètre " + i + " après mise à jour: " + param);
        }
    
        // Mettre à jour le taux d'apprentissage si nécessaire
        calculateLearningRate();
    }
    
    
    

    // Calcul du taux d'apprentissage avec warmup et décroissance
    float calculateLearningRate() {
        float step = Math.max(1.0f, (float) currentStep);  // Commencer à 1 pour éviter la division par zéro

        // Learning rate de base
        float lr = initialLr;

        // Calculer le facteur de warmup (0 à 1 pendant la période de warmup)
        float warmupFactor = Math.min(1.0f, step / warmupSteps);

        // Calculer le facteur de décroissance (diminue après la période de warmup)
        float decayFactor;
        if (step <= warmupSteps) {
            decayFactor = 1.0f;
        } else {
            // Décroissance en racine carrée inverse après le warmup
            decayFactor = (float) Math.sqrt(warmupSteps / step);
        }

        // Calculer le learning rate final
        this.learningRate = lr * warmupFactor * decayFactor;

        // Appliquer des limites pour éviter des valeurs extrêmes
        this.learningRate = Math.max(this.learningRate, initialLr * 0.1f);   // Pas moins de 10% du lr initial
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

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
}
