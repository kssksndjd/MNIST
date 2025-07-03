package CW_1;

import java.util.*;

public class HybridModel {

    // Models used in the hybrid approach
    private final KNearestNeighbor knn;
    private final MultiLayerPerceptron mlp;

    // Weights for weighted voting
    double knnWeight;
    double mlpWeight;

    // Constants for k-NN
    private static final int DEFAULT_NEIGHBOR_COUNT = 1;

    /**
     * Initializes the Hybrid Model with k-NN and MLP components.
     *
     * @param inputSize  Number of input features
     * @param hiddenSize Number of neurons in the hidden layer (MLP)
     * @param outputSize Number of output classes
     * @param k          Number of neighbors for k-NN
     * @param knnWeight  Weight for k-NN in voting
     * @param mlpWeight  Weight for MLP in voting
     */
    public HybridModel(int inputSize, int hiddenSize, int outputSize, int k, double knnWeight, double mlpWeight) {
        this.knn = new KNearestNeighbor();
        this.mlp = new MultiLayerPerceptron(inputSize, hiddenSize, outputSize, 0.012);
        this.knnWeight = knnWeight;
        this.mlpWeight = mlpWeight;
    }

    /**
     * Trains the MLP model on the provided dataset.
     *
     * @param trainData Training dataset
     * @param epochs    Number of training epochs for MLP
     */
    public void train(ArrayList<double[]> trainData, int epochs) {
        mlp.train(trainData, epochs);
    }

    /**
     * Predicts the label for a given instance using weighted voting.
     *
     * @param features   Input features of the instance
     * @param trainData  Training dataset (for k-NN)
     * @return Predicted label
     */
    public int predict(double[] features, ArrayList<double[]> trainData) {
        // Get predictions from both models
        int knnPrediction = knn.classify(features, trainData, DEFAULT_NEIGHBOR_COUNT);
        int mlpPrediction = mlp.predict(features);

        // Combine predictions using weighted voting
        double knnScore = knnPrediction * knnWeight;
        double mlpScore = mlpPrediction * mlpWeight;

        return knnScore > mlpScore ? knnPrediction : mlpPrediction;
    }

    /**
     * Evaluates the hybrid model's accuracy on the test dataset.
     *
     * @param testData   Testing dataset
     * @param trainData  Training dataset
     * @return Accuracy as a double value (between 0 and 1)
     */
    public double evaluate(ArrayList<double[]> testData, ArrayList<double[]> trainData) {
        int correctPredictions = 0;

        for (double[] instance : testData) {
            // Extract features and target label
            double[] features = Arrays.copyOf(instance, instance.length - 1);
            int targetLabel = (int) instance[instance.length - 1];

            // Predict and check correctness
            int prediction = predict(features, trainData);
            if (prediction == targetLabel) {
                correctPredictions++;
            }
        }

        return (double) correctPredictions / testData.size();
    }

}
