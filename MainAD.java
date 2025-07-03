package CW_1;

import java.io.*;
import java.util.ArrayList;

public class MainAD {

    // Paths to datasets
    private static final String DATASET1_PATH = "/Users/lyphilong/Desktop/Java/untitled copy/src/cw/dataSet1.csv";
    private static final String DATASET2_PATH = "/Users/lyphilong/Desktop/Java/untitled copy/src/cw/dataSet2.csv";
    static final int PERCENT_CONVERSION = 100;

    public static void main(String[] args) {
        try {
            // Load datasets
            ArrayList<double[]> dataSet1 = loadCSV(DATASET1_PATH);
            ArrayList<double[]> dataSet2 = loadCSV(DATASET2_PATH);

            // Normalize datasets
            normalizeData(dataSet1);
            normalizeData(dataSet2);

            // Augment the datasets
            final int AUGMENT_FACTOR = 4; // Augment with 5 new samples per instance
            //dataSet1 = AugmentData.augmentData(dataSet1, AUGMENT_FACTOR);
            //dataSet2 = AugmentData.augmentData(dataSet2, AUGMENT_FACTOR);

            // Train and evaluate Hybrid Model
            evaluateHybridModel(dataSet1, dataSet2);
        } catch (IOException e) {
            System.err.println("Error loading datasets: " + e.getMessage());
        }
    }


    /**
     * Loads data from a CSV file into an ArrayList of double arrays.
     *
     * @param filePath Path to the CSV file
     * @return Dataset as an ArrayList of double arrays
     * @throws IOException If an error occurs during file reading
     */
    private static ArrayList<double[]> loadCSV(String filePath) throws IOException {
        ArrayList<double[]> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            double[] row = new double[values.length];
            for (int featureIndex = 0; featureIndex < values.length; featureIndex++) {
                row[featureIndex] = Double.parseDouble(values[featureIndex]);
            }
            data.add(row);
        }
        br.close();
        return data;
    }

    /**
     * Normalizes the dataset features to the range [0, 1].
     * The last column (label) is excluded from normalization.
     *
     * @param data Dataset to normalize
     */
    private static void normalizeData(ArrayList<double[]> data) {
        int numFeatures = data.get(0).length - 1; // Exclude label
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];
        // Initialize min and max arrays
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            minValues[featureIndex] = Double.MAX_VALUE;
            maxValues[featureIndex] = Double.MIN_VALUE;
        }
        // Find min and max values for each feature
        for (double[] row : data) {
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                if (row[featureIndex] < minValues[featureIndex]) {
                    minValues[featureIndex] = row[featureIndex];
                }
                if (row[featureIndex] > maxValues[featureIndex]) {
                    maxValues[featureIndex] = row[featureIndex];
                }
            }
        }
        // Normalize features to [0, 1]
        for (double[] row : data) {
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                row[featureIndex] = (row[featureIndex] - minValues[featureIndex]) /
                        (maxValues[featureIndex] - minValues[featureIndex]);
            }
        }
    }


    /**
     * Evaluates the Hybrid Model on the provided datasets.
     *
     * @param trainData Training dataset
     * @param testData  Testing dataset
     */
    private static void evaluateHybridModel(ArrayList<double[]> trainData, ArrayList<double[]> testData) {
        final int INPUT_SIZE = trainData.get(0).length - 1; // Exclude label
        final int OUTPUT_SIZE = 10; // Number of classes (digits 0-9)
        final int HIDDEN_SIZE = 32; // Hidden layer size for MLP
        final int EPOCHS =400; // Training epochs for MLP
        final int K_NEIGHBORS = 1; // Number of neighbors for k-NN

        try {
            // Initialize Hybrid Model
            HybridModel hybridModel = new HybridModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, K_NEIGHBORS, 0.64, 0.36);

            // Train the model
            hybridModel.train(trainData, EPOCHS);

            // Evaluate on both folds
            double accuracy1 = hybridModel.evaluate(testData, trainData);
            double accuracy2 = hybridModel.evaluate(trainData, testData);

            // Print results
            System.out.println("Hybrid Model Accuracy (Fold 1): " + String.format("%.2f", accuracy1 * PERCENT_CONVERSION) + "%");
            System.out.println("Hybrid Model Accuracy (Fold 1): " + String.format("%.2f", accuracy2 * PERCENT_CONVERSION) + "%");
            System.out.println("Hybrid Model Average Accuracy: " + String.format("%.2f", ((accuracy1 + accuracy2) / 2) * PERCENT_CONVERSION) + "%");
        } catch (Exception error) {
            System.err.println("Error during model training or evaluation: " + error.getMessage());
        }
    }
}