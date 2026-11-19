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
    private int maxEpochs = -1;
    private float learningRate;

    // Planning de taux d'apprentissage configurable (LEGACY = comportement historique)
    public enum Schedule { LEGACY, CONSTANT, LINEAR, COSINE }
    private Schedule schedule = Schedule.LEGACY;
    private int scheduleStartStep = 0;
    private int scheduleWarmup = 0;
    private int scheduleTotalSteps = 1;
    private float scheduleMinRatio = 0f;

    /**
     * Active un planning explicite : montée linéaire sur warmupSteps puis décroissance
     * (cosinus ou linéaire) de peakLr vers peakLr*minRatio sur totalSteps. Le planning
     * est compté à partir du pas COURANT (reprise d'un checkpoint : on repart de peakLr,
     * pas de la fin de l'ancien planning). Non sérialisé dans les checkpoints : à
     * reconfigurer à chaque reprise.
     */
    public void configureSchedule(Schedule schedule, float peakLr, int warmupSteps, int totalSteps, float minRatio) {
        if (totalSteps < 1 || minRatio < 0 || minRatio > 1) {
            throw new IllegalArgumentException("totalSteps >= 1 et 0 <= minRatio <= 1 requis");
        }
        this.schedule = schedule;
        this.initialLr = peakLr;
        this.scheduleWarmup = warmupSteps;
        this.scheduleTotalSteps = totalSteps;
        this.scheduleMinRatio = minRatio;
        this.scheduleStartStep = currentStep;
        this.learningRate = learningRateAt(currentStep);
    }

    /** Taux d'apprentissage du planning explicite au pas absolu donné. */
    float learningRateAt(int absoluteStep) {
        int t = Math.max(0, absoluteStep - scheduleStartStep);
        if (scheduleWarmup > 0 && t < scheduleWarmup) {
            return initialLr * (t + 1f) / scheduleWarmup;
        }
        if (schedule == Schedule.CONSTANT) {
            return initialLr;
        }
        float span = Math.max(1, scheduleTotalSteps - scheduleWarmup);
        float prog = Math.min(1f, Math.max(0f, (t - scheduleWarmup) / span));
        float floor = initialLr * scheduleMinRatio;
        if (schedule == Schedule.LINEAR) {
            return floor + (initialLr - floor) * (1f - prog);
        }
        return floor + (initialLr - floor) * 0.5f * (1f + (float) Math.cos(Math.PI * prog));
    }

    public Schedule getSchedule() {
        return schedule;
    }

    /** Diagnostic d'un pas d'Adam : normes par paramètre (gradient reçu, paramètre avant, mise à jour appliquée). */
    public static class StepDiagnostics {
        public int step;
        public float learningRate;
        public double[] gradNorm, paramNorm, updateNorm;
        public long[][] shapes;
        /** Plancher du dénominateur : un biais initialisé à 0 a une norme ~0, ce qui rendrait le ratio absurde. */
        public static final double RATIO_FLOOR = 0.1;

        /** Plus grand ratio ||mise à jour|| / ||paramètre|| du pas et l'indice du paramètre concerné. */
        public double maxUpdateRatio() {
            double m = 0;
            for (int i = 0; i < updateNorm.length; i++) {
                m = Math.max(m, updateNorm[i] / Math.max(RATIO_FLOOR, paramNorm[i]));
            }
            return m;
        }

        public int argMaxUpdateRatio() {
            double m = -1;
            int k = 0;
            for (int i = 0; i < updateNorm.length; i++) {
                double r = updateNorm[i] / Math.max(RATIO_FLOOR, paramNorm[i]);
                if (r > m) { m = r; k = i; }
            }
            return k;
        }
    }

    private transient java.util.function.Consumer<StepDiagnostics> diagnostics;

    /** Reçoit un StepDiagnostics à chaque update (coût : quelques normes par paramètre) ; null pour désactiver. */
    public void setDiagnostics(java.util.function.Consumer<StepDiagnostics> d) {
        this.diagnostics = d;
    }

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

    /** Écrit les moments d'Adam (m, v) de chaque paramètre. */
    public void writeMoments(ObjectOutputStream oos) throws IOException {
        oos.writeInt(states.size());
        for (AdamState st : states) {
            oos.writeObject(st.m);
            oos.writeObject(st.v);
        }
    }

    /** Relit les moments d'Adam ; refuse un checkpoint dont les formes ne correspondent pas. */
    public void readMoments(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        int n = ois.readInt();
        if (n != states.size()) {
            throw new IOException("Checkpoint incompatible : " + n + " états Adam, " + states.size() + " attendus.");
        }
        for (int i = 0; i < n; i++) {
            INDArray m = (INDArray) ois.readObject();
            INDArray v = (INDArray) ois.readObject();
            if (!java.util.Arrays.equals(m.shape(), states.get(i).m.shape())) {
                throw new IOException("Checkpoint incompatible : forme Adam différente pour le paramètre " + i);
            }
            states.get(i).m.assign(m);
            states.get(i).v.assign(v);
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
        StepDiagnostics diag = null;
        if (diagnostics != null) {
            diag = new StepDiagnostics();
            diag.step = currentStep;
            diag.learningRate = learningRate;
            diag.gradNorm = new double[params.size()];
            diag.paramNorm = new double[params.size()];
            diag.updateNorm = new double[params.size()];
            diag.shapes = new long[params.size()][];
        }
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
        
            if (diag != null) {
                diag.gradNorm[i] = grad.norm2Number().doubleValue();
                diag.paramNorm[i] = param.norm2Number().doubleValue();
                diag.updateNorm[i] = step.norm2Number().doubleValue();
                diag.shapes[i] = param.shape();
            }

            // Mise à jour du paramètre
            param.subi(step);
        
            // Log après mise à jour
            // System.out.println("Paramètre " + i + " après mise à jour: " + param);
        }
    
        if (diag != null) {
            diagnostics.accept(diag);
        }

        // Mettre à jour le taux d'apprentissage si nécessaire
        calculateLearningRate();
    }
    
    
    

    // Calcul du taux d'apprentissage avec warmup et décroissance
    float calculateLearningRate() {
        if (schedule != Schedule.LEGACY) {
            this.learningRate = learningRateAt(currentStep);
            return this.learningRate;
        }
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

    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    public int getMaxEpochs() {
        return this.maxEpochs;
    }
}
