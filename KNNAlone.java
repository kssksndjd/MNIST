package cw;

import java.io.*;
import java.util.*;

public class KNNAlone {
    static final int PERCENT_CONVERSION = 100;


    // Method to load CSV data from a file
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

    // Normalize the data using min-max scaling
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

    // k-NN Classifier class
    static class KNearestNeighbor {

        // Method to classify a test instance
        public int classify(double[] testInstance, ArrayList<double[]> trainingDataset, int numNeighbors) {
            // Step 1: Calculate distances between the test instance and all training instances
            double[][] neighbors = calculateDistances(testInstance, trainingDataset);

            // Step 2: Sort the neighbors by their distance (ascending order)
            sortNeighborsByDistance(neighbors);

            // Step 3: Determine the majority label from the top k neighbors
            return getMajorityLabel(neighbors, numNeighbors);
        }

        /**
         * Calculates the distances between the test instance and all training instances.
         *
         * @param testInstance      The data point to classify.
         * @param trainingDataset   The dataset containing training instances.
         * @return                  A 2D array where each row contains [label, distance].
         */
        private double[][] calculateDistances(double[] testInstance, ArrayList<double[]> trainingDataset) {
            double[][] neighbors = new double[trainingDataset.size()][2]; // {label, distance}

            // Iterate through each training instance
            for (int trainingIndex = 0; trainingIndex < trainingDataset.size(); trainingIndex++) {
                double[] trainingInstance = trainingDataset.get(trainingIndex);

                // Calculate the Euclidean distance between test and training instances
                double distance = calculateEuclideanDistance(testInstance, trainingInstance);

                // Store the label (last value in the training instance) and distance
                neighbors[trainingIndex][0] = trainingInstance[trainingInstance.length - 1]; // label
                neighbors[trainingIndex][1] = distance; // distance
            }
            return neighbors;
        }

        /**
         * Sorts the neighbors array by distance using the bubble sort algorithm.
         *
         * @param neighbors   A 2D array where each row contains [label, distance].
         */
        private void sortNeighborsByDistance(double[][] neighbors) {
            // Perform bubble sort to arrange neighbors in ascending order of distance
            for (int pass = 0; pass < neighbors.length - 1; pass++) {
                for (int currentIndex = 0; currentIndex < neighbors.length - pass - 1; currentIndex++) {
                    // Compare the distances of adjacent neighbors
                    if (neighbors[currentIndex][1] > neighbors[currentIndex + 1][1]) {
                        // Swap the rows if they are out of order
                        double[] temp = neighbors[currentIndex];
                        neighbors[currentIndex] = neighbors[currentIndex + 1];
                        neighbors[currentIndex + 1] = temp;
                    }
                }
            }
        }

        /**
         * Finds the majority label from the top k neighbors using majority voting.
         *
         * @param neighbors      A 2D array where each row contains [label, distance].
         * @param numNeighbors   The number of nearest neighbors to consider (k).
         * @return               The label with the highest occurrence in the top k neighbors.
         */
        private int getMajorityLabel(double[][] neighbors, int numNeighbors) {
            int[] labelCounts = new int[256]; // Array to count occurrences of each label (adjust size as needed)

            // Count the labels in the top k neighbors
            for (int neighborIndex = 0; neighborIndex < numNeighbors; neighborIndex++) {
                int label = (int) neighbors[neighborIndex][0]; // Get the label of the neighbor
                labelCounts[label]++; // Increment the count for this label
            }

            // Find the label with the highest count
            int majorityLabel = -1; // Variable to store the majority label
            int maxCount = -1; // Variable to store the highest count found
            for (int labelIndex = 0; labelIndex < labelCounts.length; labelIndex++) {
                if (labelCounts[labelIndex] > maxCount) {
                    maxCount = labelCounts[labelIndex]; // Update the maximum count
                    majorityLabel = labelIndex; // Update the majority label
                }
            }
            return majorityLabel;
        }

        /**
         * Calculates the Euclidean distance between two data points.
         *
         * @param instance1   The first data point.
         * @param instance2   The second data point.
         * @return            The Euclidean distance between the two points.
         */



        // Calculate Euclidean distance between two instances
        private double calculateEuclideanDistance(double[] instance1, double[] instance2) {
            double sumOfSquares = 0;
            for (int featureIndex = 0; featureIndex < instance1.length - 1; featureIndex++) { // Exclude the label
                sumOfSquares += Math.pow(instance1[featureIndex] - instance2[featureIndex], 2);
            }
            return Math.sqrt(sumOfSquares);
        }
    }

    // Run k-NN on test data and return the accuracy
    public static double evaluateKnnModel(ArrayList<double[]> trainingDataset, ArrayList<double[]> testingDataset, int numNeighbors) {
        KNearestNeighbor knnClassifier = new KNearestNeighbor();
        int correctPredictions = 0;

        for (double[] testInstance : testingDataset) {
            int predictedLabel = knnClassifier.classify(testInstance, trainingDataset, numNeighbors);
            if (predictedLabel == testInstance[testInstance.length - 1]) {
                correctPredictions++;
            }
        }

        return (double) correctPredictions / testingDataset.size();
    }

    // Method to perform two-fold cross-validation
    public static void performTwoFoldValidation(ArrayList<double[]> dataset1, ArrayList<double[]> dataset2, int numNeighbors) {
        System.out.println("Performing Two-Fold Cross-Validation with k=" + numNeighbors);

        // First round: train on dataset1, test on dataset2
        double accuracy1 = evaluateKnnModel(dataset1, dataset2, numNeighbors);
        System.out.println("Accuracy (Train on dataset1, Test on dataset2): " + String.format("%.2f", accuracy1 * PERCENT_CONVERSION) + "%");

        // Second round: train on dataset2, test on dataset1
        double accuracy2 = evaluateKnnModel(dataset2, dataset1, numNeighbors);
        System.out.println("Accuracy (Train on dataset2, Test on dataset1): " + String.format("%.2f", accuracy2 * PERCENT_CONVERSION) + "%");

        // Average accuracy
        double averageAccuracy = (accuracy1 + accuracy2) / 2;
        System.out.println("Average Accuracy: " + String.format("%.2f", averageAccuracy * PERCENT_CONVERSION) + "%");
    }

    // Main method to execute the program
    public static void main(String[] args) {
        try {
            final int PERCENT_CONVERSION = 100;
            // Load datasets
            String dataset1Path = "/Users/lyphilong/Desktop/Java/untitled copy/src/cw/dataSet1.csv";
            String dataset2Path = "/Users/lyphilong/Desktop/Java/untitled copy/src/cw/dataSet2.csv";
            ArrayList<double[]> dataset1 = loadCSV(dataset1Path);
            ArrayList<double[]> dataset2 = loadCSV(dataset2Path);

            // Normalize datasets
            //normalizeData(dataset1);
            //normalizeData(dataset2);

            // Perform two-fold cross-validation with k-NN
            int numNeighbors = 1; // Number of neighbors (can be tuned)
            performTwoFoldValidation(dataset1, dataset2, numNeighbors);

        } catch (IOException exception) {
            System.err.println("Error loading the data: " + exception.getMessage());
        }
    }
}
