package cw;

import java.io.*;
import java.util.*;

public class MLPAlone {
    static final int PERCENT_CONVERSION = 100;
    private final int numInputNeurons;  // Number of input neurons
    private final int numHiddenNeurons; // Number of hidden neurons
    private final int numOutputNeurons; // Number of output neurons
    private final double learningRate;  // Learning rate

    private final double[][] inputToHiddenWeights; // Weights from input to hidden layer
    private final double[][] hiddenToOutputWeights; // Weights from hidden to output layer
    private final double[] hiddenLayerBiases; // Biases for hidden layer
    private final double[] outputLayerBiases; // Biases for output layer

    private final Random randomGenerator; // Random generator for initializing weights

    // Constructor
    public MLPAlone(int numInputNeurons, int numHiddenNeurons, int numOutputNeurons, double learningRate) {
        this.numInputNeurons = numInputNeurons;
        this.numHiddenNeurons = numHiddenNeurons;
        this.numOutputNeurons = numOutputNeurons;
        this.learningRate = learningRate;

        // Initialize weights and biases
        this.inputToHiddenWeights = new double[numInputNeurons][numHiddenNeurons];
        this.hiddenToOutputWeights = new double[numHiddenNeurons][numOutputNeurons];
        this.hiddenLayerBiases = new double[numHiddenNeurons];
        this.outputLayerBiases = new double[numOutputNeurons];

        this.randomGenerator = new Random();
        for (int input = 0; input < numInputNeurons; input++) {
            for (int hidden = 0; hidden < numHiddenNeurons; hidden++) {
                inputToHiddenWeights[input][hidden] = randomGenerator.nextGaussian() * 0.01;
            }
        }
        for (int hidden = 0; hidden < numHiddenNeurons; hidden++) {
            for (int output = 0; output < numOutputNeurons; output++) {
                hiddenToOutputWeights[hidden][output] = randomGenerator.nextGaussian() * 0.01;
            }
        }
        Arrays.fill(hiddenLayerBiases, 0);
        Arrays.fill(outputLayerBiases, 0);
    }

    // Sigmoid activation function
    private double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    // Softmax activation function
    private double[] softmax(double[] values) {
        double max = Arrays.stream(values).max().getAsDouble();
        double sum = 0.0;
        double[] probabilities = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            probabilities[i] = Math.exp(values[i] - max);
            sum += probabilities[i];
        }
        for (int i = 0; i < values.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }

    // Forward pass
    public double[] forwardPass(double[] inputFeatures) {
        double[] hiddenLayerOutputs = new double[numHiddenNeurons];
        for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
            hiddenLayerOutputs[hiddenNeuron] = hiddenLayerBiases[hiddenNeuron];
            for (int inputNeuron = 0; inputNeuron < numInputNeurons; inputNeuron++) {
                hiddenLayerOutputs[hiddenNeuron] += inputFeatures[inputNeuron] * inputToHiddenWeights[inputNeuron][hiddenNeuron];
            }
            hiddenLayerOutputs[hiddenNeuron] = sigmoid(hiddenLayerOutputs[hiddenNeuron]);
        }

        double[] outputLayerOutputs = new double[numOutputNeurons];
        for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
            outputLayerOutputs[outputNeuron] = outputLayerBiases[outputNeuron];
            for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
                outputLayerOutputs[outputNeuron] += hiddenLayerOutputs[hiddenNeuron] * hiddenToOutputWeights[hiddenNeuron][outputNeuron];
            }
        }
        return softmax(outputLayerOutputs);
    }

    // Backpropagation and weight updates
    public void train(double[] inputFeatures, int trueLabel) {
        // Step 1: Forward pass
        double[] hiddenLayerOutputs = calculateHiddenLayerOutputs(inputFeatures);
        double[] outputLayerOutputs = calculateOutputLayerOutputs(hiddenLayerOutputs);
        double[] predictedProbabilities = softmax(outputLayerOutputs);

        // Step 2: Backward pass (calculate errors)
        double[] outputLayerErrors = calculateOutputLayerErrors(predictedProbabilities, trueLabel);
        double[] hiddenLayerErrors = calculateHiddenLayerErrors(hiddenLayerOutputs, outputLayerErrors);

        // Step 3: Update weights and biases
        updateOutputLayer(hiddenLayerOutputs, outputLayerErrors);
        updateHiddenLayer(inputFeatures, hiddenLayerErrors);
    }

    /**
     * Calculates the activations for the hidden layer using input features.
     */
    private double[] calculateHiddenLayerOutputs(double[] inputFeatures) {
        double[] hiddenLayerOutputs = new double[numHiddenNeurons];
        for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
            hiddenLayerOutputs[hiddenNeuron] = hiddenLayerBiases[hiddenNeuron];
            for (int inputNeuron = 0; inputNeuron < numInputNeurons; inputNeuron++) {
                hiddenLayerOutputs[hiddenNeuron] += inputFeatures[inputNeuron] * inputToHiddenWeights[inputNeuron][hiddenNeuron];
            }
            hiddenLayerOutputs[hiddenNeuron] = sigmoid(hiddenLayerOutputs[hiddenNeuron]);
        }
        return hiddenLayerOutputs;
    }

    /**
     * Calculates the activations for the output layer using hidden layer outputs.
     */
    private double[] calculateOutputLayerOutputs(double[] hiddenLayerOutputs) {
        double[] outputLayerOutputs = new double[numOutputNeurons];
        for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
            outputLayerOutputs[outputNeuron] = outputLayerBiases[outputNeuron];
            for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
                outputLayerOutputs[outputNeuron] += hiddenLayerOutputs[hiddenNeuron] * hiddenToOutputWeights[hiddenNeuron][outputNeuron];
            }
        }
        return outputLayerOutputs;
    }

    /**
     * Calculates the error at the output layer.
     */
    private double[] calculateOutputLayerErrors(double[] predictedProbabilities, int trueLabel) {
        double[] outputLayerErrors = new double[numOutputNeurons];
        for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
            outputLayerErrors[outputNeuron] = predictedProbabilities[outputNeuron] - (outputNeuron == trueLabel ? 1.0 : 0.0);
        }
        return outputLayerErrors;
    }

    /**
     * Calculates the error at the hidden layer.
     */
    private double[] calculateHiddenLayerErrors(double[] hiddenLayerOutputs, double[] outputLayerErrors) {
        double[] hiddenLayerErrors = new double[numHiddenNeurons];
        for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
            hiddenLayerErrors[hiddenNeuron] = 0.0;
            for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
                hiddenLayerErrors[hiddenNeuron] += outputLayerErrors[outputNeuron] * hiddenToOutputWeights[hiddenNeuron][outputNeuron];
            }
            hiddenLayerErrors[hiddenNeuron] *= hiddenLayerOutputs[hiddenNeuron] * (1.0 - hiddenLayerOutputs[hiddenNeuron]);
        }
        return hiddenLayerErrors;
    }

    /**
     * Updates the weights and biases for the output layer using errors.
     */
    private void updateOutputLayer(double[] hiddenLayerOutputs, double[] outputLayerErrors) {
        for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {
            for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
                hiddenToOutputWeights[hiddenNeuron][outputNeuron] -= learningRate * outputLayerErrors[outputNeuron] * hiddenLayerOutputs[hiddenNeuron];
            }
            outputLayerBiases[outputNeuron] -= learningRate * outputLayerErrors[outputNeuron];
        }
    }

    /**
     * Updates the weights and biases for the hidden layer using errors.
     */
    private void updateHiddenLayer(double[] inputFeatures, double[] hiddenLayerErrors) {
        for (int hiddenNeuron = 0; hiddenNeuron < numHiddenNeurons; hiddenNeuron++) {
            for (int inputNeuron = 0; inputNeuron < numInputNeurons; inputNeuron++) {
                inputToHiddenWeights[inputNeuron][hiddenNeuron] -= learningRate * hiddenLayerErrors[hiddenNeuron] * inputFeatures[inputNeuron];
            }
            hiddenLayerBiases[hiddenNeuron] -= learningRate * hiddenLayerErrors[hiddenNeuron];
        }
    }

    // Accuracy evaluation
    public double evaluateAccuracy(ArrayList<double[]> dataset) {
        int correctPredictions = 0;
        for (double[] instance : dataset) {
            double[] inputFeatures = Arrays.copyOf(instance, numInputNeurons);
            int trueLabel = (int) instance[numInputNeurons];
            int predictedLabel = predict(inputFeatures);
            if (predictedLabel == trueLabel) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / dataset.size();
    }

    // Prediction method
    public int predict(double[] inputFeatures) {
        double[] outputProbabilities = forwardPass(inputFeatures);
        int predictedClass = 0;
        for (int outputNeuron = 1; outputNeuron < outputProbabilities.length; outputNeuron++) {
            if (outputProbabilities[outputNeuron] > outputProbabilities[predictedClass]) {
                predictedClass = outputNeuron;
            }
        }
        return predictedClass;
    }
    public static ArrayList<double[]> loadCSV(String filePath) throws IOException {
        ArrayList<double[]> dataRows = new ArrayList<>();
        BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));
        String currentLine;
        while ((currentLine = bufferedReader.readLine()) != null) {
            String[] stringValues = currentLine.split(",");
            double[] numericValues = new double[stringValues.length];
            for (int columnIndex = 0; columnIndex < stringValues.length; columnIndex++) {
                numericValues[columnIndex] = Double.parseDouble(stringValues[columnIndex]);
            }
            dataRows.add(numericValues);
        }
        bufferedReader.close();
        return dataRows;
    }
    public static void normalizeData(ArrayList<double[]> dataset) {
        int numFeatures = dataset.get(0).length - 1; // Assuming the last column is the label
        double[] featureMinValues = new double[numFeatures];
        double[] featureMaxValues = new double[numFeatures];
        Arrays.fill(featureMinValues, Double.MAX_VALUE);
        Arrays.fill(featureMaxValues, Double.MIN_VALUE);

        // Find min and max for each feature
        for (double[] dataRow : dataset) {
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                if (dataRow[featureIndex] < featureMinValues[featureIndex]) {
                    featureMinValues[featureIndex] = dataRow[featureIndex];
                }
                if (dataRow[featureIndex] > featureMaxValues[featureIndex]) {
                    featureMaxValues[featureIndex] = dataRow[featureIndex];
                }
            }
        }

        // Apply min-max normalization
        for (double[] dataRow : dataset) {
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                dataRow[featureIndex] =
                        (dataRow[featureIndex] - featureMinValues[featureIndex]) /
                                (featureMaxValues[featureIndex] - featureMinValues[featureIndex]);
            }
        }
    }


    public static void main(String[] args) throws IOException {
        // Paths to the datasets
        String path1 = "/Users/lyphilong/Desktop/Java/untitled/src/cw/dataSet1.csv";
        String path2 = "/Users/lyphilong/Desktop/Java/untitled/src/cw/dataSet2.csv";

        // Load and normalize datasets
        ArrayList<double[]> trainingSet = loadCSV(path1);
        ArrayList<double[]> testingSet = loadCSV(path2);

        normalizeData(trainingSet);
        normalizeData(testingSet);

        // Network configuration
        int numInputNeurons = trainingSet.get(0).length - 1; // Exclude the label
        int numHiddenNeurons = 32; // Adjustable hyperparameter
        int numOutputNeurons = 10; // Number of classes (0-9)
        double learningRate = 0.01;

        // Instantiate the MLP
        MLPAlone mlp = new MLPAlone(numInputNeurons, numHiddenNeurons, numOutputNeurons, learningRate);

        // Train on the first dataset and test on the second
        for (int epoch = 0; epoch < 100; epoch++) {
            for (double[] instance : trainingSet) {
                double[] inputFeatures = Arrays.copyOf(instance, numInputNeurons);
                int trueLabel = (int) instance[numInputNeurons];
                mlp.train(inputFeatures, trueLabel);
            }
        }
        double accuracyFold1 = mlp.evaluateAccuracy(testingSet);
        System.out.println("Fold 1 Accuracy: " + String.format("%.2f", accuracyFold1 * PERCENT_CONVERSION) + "%");
        // Train on the second dataset and test on the first
        mlp = new MLPAlone(numInputNeurons, numHiddenNeurons, numOutputNeurons, learningRate);
        for (int epoch = 0; epoch < 100; epoch++) {
            for (double[] instance : testingSet) {
                double[] inputFeatures = Arrays.copyOf(instance, numInputNeurons);
                int trueLabel = (int) instance[numInputNeurons];
                mlp.train(inputFeatures, trueLabel);
            }
        }
        double accuracyFold2 = mlp.evaluateAccuracy(trainingSet);
        System.out.println("Fold 2 Accuracy: " + String.format("%.2f", accuracyFold2 * PERCENT_CONVERSION) + "%");

        // Calculate and print average accuracy
        double averageAccuracy = (accuracyFold1 + accuracyFold2) / 2;
        System.out.println("Average Accuracy: " + String.format("%.2f", averageAccuracy * PERCENT_CONVERSION) + "%");
    }

}
