package CW_1;

import java.util.*;

public class MultiLayerPerceptron {

    private final int inputSize;          // Number of input features
    private final int hiddenLayerSize;   // Number of neurons in the hidden layer
    private final int outputSize;        // Number of output classes
    private final double learningRate;   // Learning rate for weight updates

    private final double[][] inputToHiddenWeights;  // Weights between input and hidden layer
    private final double[][] hiddenToOutputWeights; // Weights between hidden and output layer
    private final double[] hiddenLayerBiases;       // Biases for the hidden layer
    private final double[] outputLayerBiases;       // Biases for the output layer

    private final Random random;        // Random generator for initializing weights

    private static final double WEIGHT_INIT_STD_DEV = 0.01; // Standard deviation for weight initialization

    /**
     * Constructor to initialize the MLP model.
     *
     * @param inputSize        Number of input features.
     * @param hiddenLayerSize  Number of neurons in the hidden layer.
     * @param outputSize       Number of output classes.
     * @param learningRate     Learning rate for gradient descent.
     */
    public MultiLayerPerceptron(int inputSize, int hiddenLayerSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        // Initialize weights and biases
        this.inputToHiddenWeights = new double[inputSize][hiddenLayerSize];
        this.hiddenToOutputWeights = new double[hiddenLayerSize][outputSize];
        this.hiddenLayerBiases = new double[hiddenLayerSize];
        this.outputLayerBiases = new double[outputSize];

        this.random = new Random();
        initializeWeightsAndBiases();
    }

    /**
     * Initializes weights and biases with small random values.
     */
    private void initializeWeightsAndBiases() {
        for (int inputNeuron = 0; inputNeuron < inputSize; inputNeuron++) {
            for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
                inputToHiddenWeights[inputNeuron][hiddenNeuron] = random.nextGaussian() * WEIGHT_INIT_STD_DEV;
            }
        }

        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
            for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
                hiddenToOutputWeights[hiddenNeuron][outputNeuron] = random.nextGaussian() * WEIGHT_INIT_STD_DEV;
            }
        }
    }

    /**
     * Trains the MLP using the given training data for the specified number of epochs.
     *
     * @param trainingData Training dataset.
     * @param epochs       Number of epochs.
     */
    public void train(List<double[]> trainingData, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (double[] dataPoint : trainingData) {
                double[] features = Arrays.copyOf(dataPoint, inputSize);
                int targetLabel = (int) dataPoint[inputSize];

                // Forward pass
                double[] hiddenLayerOutputs = computeHiddenLayerOutputs(features);
                double[] outputLayerOutputs = computeOutputLayerOutputs(hiddenLayerOutputs);

                // Backpropagation
                double[] outputErrors = computeOutputErrors(outputLayerOutputs, targetLabel);
                double[] hiddenErrors = computeHiddenErrors(outputErrors, hiddenLayerOutputs);

                // Update weights and biases
                updateWeightsAndBiases(features, hiddenLayerOutputs, outputErrors, hiddenErrors);
            }
        }
    }

    /**
     * Predicts the class label for a given input instance.
     *
     * @param features Input features.
     * @return Predicted class label.
     */
    public int predict(double[] features) {
        double[] hiddenLayerOutputs = computeHiddenLayerOutputs(features);
        double[] outputLayerOutputs = computeOutputLayerOutputs(hiddenLayerOutputs);
        return argMax(outputLayerOutputs);
    }

    /**
     * Performs forward pass to compute hidden layer outputs.
     *
     * @param inputs Input features.
     * @return Activations of the hidden layer.
     */
    private double[] computeHiddenLayerOutputs(double[] inputs) {
        double[] activations = new double[hiddenLayerSize];
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
            activations[hiddenNeuron] = hiddenLayerBiases[hiddenNeuron];
            for (int inputNeuron = 0; inputNeuron < inputSize; inputNeuron++) {
                activations[hiddenNeuron] += inputs[inputNeuron] * inputToHiddenWeights[inputNeuron][hiddenNeuron];
            }
            activations[hiddenNeuron] = relu(activations[hiddenNeuron]);
        }
        return activations;
    }

    /**
     * Performs forward pass to compute output layer outputs.
     *
     * @param hiddenLayerOutputs Outputs from the hidden layer.
     * @return Activations of the output layer.
     */
    private double[] computeOutputLayerOutputs(double[] hiddenLayerOutputs) {
        double[] activations = new double[outputSize];
        for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
            activations[outputNeuron] = outputLayerBiases[outputNeuron];
            for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
                activations[outputNeuron] += hiddenLayerOutputs[hiddenNeuron] * hiddenToOutputWeights[hiddenNeuron][outputNeuron];
            }
        }
        return softmax(activations);
    }

    /**
     * Computes the error at the output layer.
     *
     * @param predictedOutputs Predicted outputs from the model.
     * @param targetLabel       Correct class label.
     * @return Error values for the output layer.
     */
    private double[] computeOutputErrors(double[] predictedOutputs, int targetLabel) {
        double[] errors = new double[outputSize];
        for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
            errors[outputNeuron] = predictedOutputs[outputNeuron] - (outputNeuron == targetLabel ? 1.0 : 0.0);
        }
        return errors;
    }

    /**
     * Computes the error at the hidden layer.
     *
     * @param outputErrors       Errors at the output layer.
     * @param hiddenLayerOutputs Outputs from the hidden layer.
     * @return Error values for the hidden layer.
     */
    private double[] computeHiddenErrors(double[] outputErrors, double[] hiddenLayerOutputs) {
        double[] errors = new double[hiddenLayerSize];
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
            errors[hiddenNeuron] = 0.0;
            for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
                errors[hiddenNeuron] += outputErrors[outputNeuron] * hiddenToOutputWeights[hiddenNeuron][outputNeuron];
            }
            errors[hiddenNeuron] *= reluDerivative(hiddenLayerOutputs[hiddenNeuron]);
        }
        return errors;
    }

    /**
     * Updates weights and biases using backpropagation errors.
     *
     * @param inputs               Input features.
     * @param hiddenLayerOutputs   Outputs from the hidden layer.
     * @param outputErrors         Errors at the output layer.
     * @param hiddenErrors         Errors at the hidden layer.
     */
    private void updateWeightsAndBiases(double[] inputs, double[] hiddenLayerOutputs, double[] outputErrors, double[] hiddenErrors) {
        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
            for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
                hiddenToOutputWeights[hiddenNeuron][outputNeuron] -= learningRate * outputErrors[outputNeuron] * hiddenLayerOutputs[hiddenNeuron];
            }
        }

        for (int inputNeuron = 0; inputNeuron < inputSize; inputNeuron++) {
            for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
                inputToHiddenWeights[inputNeuron][hiddenNeuron] -= learningRate * hiddenErrors[hiddenNeuron] * inputs[inputNeuron];
            }
        }

        for (int hiddenNeuron = 0; hiddenNeuron < hiddenLayerSize; hiddenNeuron++) {
            hiddenLayerBiases[hiddenNeuron] -= learningRate * hiddenErrors[hiddenNeuron];
        }

        for (int outputNeuron = 0; outputNeuron < outputSize; outputNeuron++) {
            outputLayerBiases[outputNeuron] -= learningRate * outputErrors[outputNeuron];
        }
    }

    // Activation functions
    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    private double[] softmax(double[] inputArray) {
        double maxElement = Arrays.stream(inputArray).max().orElse(0);
        double sumOfExponentials = 0.0;
        double[] softmaxValues = new double[inputArray.length];

        // Compute the exponentials and their sum
        for (int index = 0; index < inputArray.length; index++) {
            softmaxValues[index] = Math.exp(inputArray[index] - maxElement);
            sumOfExponentials += softmaxValues[index];
        }

        // Normalize the values to get probabilities
        for (int index = 0; index < inputArray.length; index++) {
            softmaxValues[index] /= sumOfExponentials;
        }

        return softmaxValues;
    }

    // Helper method to find the index of the maximum value
    private int argMax(double[] inputArray) {
        int indexOfMaxValue = 0;
        for (int currentIndex = 1; currentIndex < inputArray.length; currentIndex++) {
            if (inputArray[currentIndex] > inputArray[indexOfMaxValue]) {
                indexOfMaxValue = currentIndex;
            }
        }
        return indexOfMaxValue;
    }
}
