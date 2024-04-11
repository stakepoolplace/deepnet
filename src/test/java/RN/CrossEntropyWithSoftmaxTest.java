package RN;

public class CrossEntropyWithSoftmaxTest {

	    public static void main(String[] args) {

	        // Input values
	        double[] inputs = {1.0, 2.0, 3.0};

	        // Weights
	        double[] weights = {0.5, 0.5, 0.5};

	        // Biases
	        double[] biases = {0.0, 0.0, 0.0};

	        // Target values
	        double[] targets = {0.3, 0.5, 0.2};

	        // Learning rate
	        double learningRate = 0.1;

	        // Number of epochs
	        int numEpochs = 10;

	        // Pre-calculated validation checks
	        double[] validationInputs = {0.5, 0.5, 0.5};
	        double[] validationTargets = {0.5, 0.5, 0.5};
	        double[] validationOutputs = forwardPropagation(validationInputs, weights, biases);
	        double validationLoss = crossEntropyLoss(validationOutputs, validationTargets);

	        for (int epoch = 0; epoch < numEpochs; epoch++) {

	            // Forward propagation
	            double[] outputs = forwardPropagation(inputs, weights, biases);

	            // Calculate cross-entropy loss
	            double loss = crossEntropyLoss(outputs, targets);

	            // Backward propagation
	            double[] gradients = backwardPropagation(outputs, targets, weights);

	            // Update weights and biases
	            weights = updateWeights(weights, gradients, learningRate);
	            biases = updateBiases(biases, gradients, learningRate);
	            

	            // Validate updated weights and biases
	            if (validateWeights(weights) && validateBiases(biases)) {
	                System.out.println("Epoch " + (epoch + 1) + " completed successfully.");
	            } else {
	                System.out.println("Invalid weights or biases detected at epoch " + (epoch + 1) + ".");
	                break;
	            }

	            // Check for convergence
	            validationOutputs = forwardPropagation(validationInputs, weights, biases);
	            double newValidationLoss = crossEntropyLoss(validationOutputs, validationTargets);
	            if (Math.abs(newValidationLoss - validationLoss) < 0.001) {
	                System.out.println("Convergence detected at epoch " + (epoch + 1) + ".");
	                break;
	            } else {
	                validationLoss = newValidationLoss;
	            }
	        }
	    }

	    // Forward propagation function
	    public static double[] forwardPropagation(double[] inputs, double[] weights, double[] biases) {
	        double[] weightedSums = new double[inputs.length];
	        for (int i = 0; i < inputs.length; i++) {
	            weightedSums[i] = inputs[i] * weights[i] + biases[i];
	        }
	        double[] outputs = new double[inputs.length];
	        for (int i = 0; i < inputs.length; i++) {
	            outputs[i] = sigmoid(weightedSums[i]);
	        }
	        return outputs;
	    }

	    // Sigmoid function
	    public static double sigmoid(double x) {
	        return 1 / (1 + Math.exp(-x));
	    }

	    // Cross-entropy loss function
	    public static double crossEntropyLoss(double[] outputs, double[] targets) {
	        double loss = 0.0;
	        for (int i = 0; i < outputs.length; i++) {
	            loss += targets[i] * Math.log(outputs[i]);
	        }
	        return -loss;
	    }

	    // Backward propagation function
	    public static double[] backwardPropagation(double[] outputs, double[] targets, double[] weights) {
	        double[] gradients = new double[outputs.length];
	        for (int i = 0; i < outputs.length; i++) {
	            gradients[i] = (outputs[i] - targets[i]) * sigmoidDerivative(outputs[i]);
	        }
	        return gradients;
	    }

	    // Sigmoid derivative function
	    public static double sigmoidDerivative(double x) {
	        double fx = sigmoid(x);
	        return fx * (1 - fx);
	    }

	    // Update weights function
	    public static double[] updateWeights(double[] weights, double[] gradients, double learningRate) {
	        double[] updatedWeights = new double[weights.length];
	        for (int i = 0; i < weights.length; i++) {
	            updatedWeights[i] = weights[i] - learningRate * gradients[i];
	        }
	        return updatedWeights;
	    }

	    // Update biases function
	    public static double[] updateBiases(double[] biases, double[] gradients, double learningRate) {
	        double[] updatedBiases = new double[biases.length];
	        for (int i = 0; i < biases.length; i++) {
	            updatedBiases[i] = biases[i] - learningRate * gradients[i];
	        }
	        return updatedBiases;
	    }

	    // Validate weights function
	    public static boolean validateWeights(double[] weights) {
	        for (double weight : weights) {
	            if (weight < 0.0 || weight > 1.0) {
	                return false;
	            }
	        }
	        return true;
	    }

	    // Validate biases function
	    public static boolean validateBiases(double[] biases) {
	        for (double bias : biases) {
	            if (bias < 0.0 || bias > 1.0) {
	                return false;
	            }
	        }
	        return true;
	    }
	}