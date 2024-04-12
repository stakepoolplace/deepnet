package RN;

public class NeuralNetwork {
    private double[][] weightsHidden; // Poids entre l'entrée et la couche cachée
    private double[] weightsOutput; // Poids entre la couche cachée et la sortie
    private double[] outputHidden; // Sorties de la couche cachée
    private double[] outputFinal; // Sorties finales du réseau pour softmax
    private double learningRate = 0.1; // Taux d'apprentissage

    public NeuralNetwork() {
        // Initialisation des poids (pour simplifier, les valeurs sont prédéfinies ici)
        weightsHidden = new double[][]{{0.15, 0.25}, {0.20, 0.30}};
        weightsOutput = new double[]{0.40, 0.50};
        outputHidden = new double[2];
        outputFinal = new double[2]; // Assumant deux classes pour softmax
    }

    public NeuralNetwork(double[][][] weightsCopy, double[][] biasesCopy, double learningRate2) {
        weightsHidden = new double[][]{{0.15, 0.25}, {0.20, 0.30}};
        weightsOutput = new double[]{0.40, 0.50};
        outputHidden = new double[2];
        outputFinal = new double[2];
	}

	public double[] forwardPass(double[] inputs) {
        // Calcul des sorties pour la couche cachée
        for (int i = 0; i < outputHidden.length; i++) {
            outputHidden[i] = sigmoid(inputs[0] * weightsHidden[0][i] + inputs[1] * weightsHidden[1][i]);
        }

        // Calcul de la sortie finale en utilisant softmax
        outputFinal = softmax(outputHidden);
        return outputFinal;
    }

    public double[] softmax(double[] inputs) {
        double sumExp = 0;
        double[] exps = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            exps[i] = Math.exp(inputs[i]);
            sumExp += exps[i];
        }
        for (int i = 0; i < inputs.length; i++) {
            exps[i] /= sumExp;
        }
        return exps;
    }

    public double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void backwardPass(double[] expected) {
        double[] error = new double[outputFinal.length];
        for (int i = 0; i < outputFinal.length; i++) {
            error[i] = expected[i] - outputFinal[i];
        }

        // Mise à jour des poids de la couche de sortie
        for (int i = 0; i < weightsOutput.length; i++) {
            for (int j = 0; j < outputFinal.length; j++) {
                weightsOutput[j] += learningRate * error[j] * outputHidden[i];
            }
        }

        // Mise à jour des poids de la couche cachée
        for (int i = 0; i < weightsHidden.length; i++) {
            for (int j = 0; j < weightsHidden[i].length; j++) {
                double inputGradient = 0;
                for (int k = 0; k < outputFinal.length; k++) {
                    inputGradient += error[k] * weightsOutput[k] * outputHidden[j] * (1 - outputHidden[j]);
                }
                weightsHidden[i][j] += learningRate * inputGradient * (i == 0 ? 1.0 : 0.0); // Simplifié pour des entrées binaires
            }
        }
    }

    public double[] getWeightsOutput() {
        return weightsOutput;
    }

    public double[][] getWeightsHidden() {
        return weightsHidden;
    }
}
