package CW_1;

import java.util.ArrayList;
public class KNearestNeighbor {

    /**
     * Classifies a test instance using the k-Nearest Neighbors algorithm.
     *
     * @param testInstance   Test instance to classify
     * @param trainData      Training dataset
     * @param neighborCount  Number of nearest neighbors to consider (k)
     * @return Predicted class label
     */
    public int classify(double[] testInstance, ArrayList<double[]> trainData, int neighborCount) {
        // Array to store distances and labels
        double[][] distances = calculateDistances(testInstance, trainData);

        // Sort distances manually
        sortDistances(distances);

        // Count votes for the top k neighbors
        int[] labelCounts = new int[256]; // Adjust size based on label range
        for (int neighborIndex = 0; neighborIndex < neighborCount; neighborIndex++) {
            int label = (int) distances[neighborIndex][0]; // Convert label to integer
            labelCounts[label]++;
        }

        // Determine the majority label
        return findMajorityLabel(labelCounts);
    }

    /**
     * Calculates distances between the test instance and all training instances.
     *
     * @param testInstance Test instance
     * @param trainData    Training dataset
     * @return Array of distances and labels
     */
    private double[][] calculateDistances(double[] testInstance, ArrayList<double[]> trainData) {
        double[][] distances = new double[trainData.size()][2];

        for (int trainIndex = 0; trainIndex < trainData.size(); trainIndex++) {
            double[] trainInstance = trainData.get(trainIndex);
            double distance = calculateEuclideanDistance(testInstance, trainInstance);
            distances[trainIndex][0] = trainInstance[trainInstance.length - 1]; // label
            distances[trainIndex][1] = distance; // distance
        }

        return distances;
    }

    /**
     * Manually sorts the distances array by the distance values.
     *
     * @param distances Array of distances and labels
     */
    private void sortDistances(double[][] distances) {
        for (int passIndex = 0; passIndex < distances.length - 1; passIndex++) {
            for (int compareIndex = 0; compareIndex < distances.length - passIndex - 1; compareIndex++) {
                if (distances[compareIndex][1] > distances[compareIndex + 1][1]) {
                    // Swap rows
                    double[] temp = distances[compareIndex];
                    distances[compareIndex] = distances[compareIndex + 1];
                    distances[compareIndex + 1] = temp;
                }
            }
        }
    }

    /**
     * Finds the label with the highest count.
     *
     * @param labelCounts Array of label counts
     * @return Label with the majority count
     */
    private int findMajorityLabel(int[] labelCounts) {
        int majorityLabel = -1;
        int maxCount = -1;

        for (int labelIndex = 0; labelIndex < labelCounts.length; labelIndex++) {
            if (labelCounts[labelIndex] > maxCount) {
                maxCount = labelCounts[labelIndex];
                majorityLabel = labelIndex;
            }
        }

        return majorityLabel;
    }

    /**
     * Calculates the Euclidean distance between two instances.
     *
     * @param instance1 First instance
     * @param instance2 Second instance
     * @return Euclidean distance
     */
    private double calculateEuclideanDistance(double[] instance1, double[] instance2) {
        double sumOfSquares = 0;

        for (int featureIndex = 0; featureIndex < instance1.length - 1; featureIndex++) { // Exclude label
            sumOfSquares += Math.pow(instance1[featureIndex] - instance2[featureIndex], 2);
        }

        return Math.sqrt(sumOfSquares);
    }
}
