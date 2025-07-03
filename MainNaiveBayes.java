package cw;



import java.io.*;
import java.util.*;

public class MainNaiveBayes {
    static final int PERCENT_CONVERSION = 100;
    private static final String DATASET1_PATH = "/Users/lyphilong/Desktop/Java/untitled/src/cw/dataSet1.csv";
    private static final String DATASET2_PATH = "/Users/lyphilong/Desktop/Java/untitled/src/cw/dataSet2.csv";

    public static void main(String[] args) throws IOException {
        ArrayList<double[]> dataSet1 = loadCSV(DATASET1_PATH);
        ArrayList<double[]> dataSet2 = loadCSV(DATASET2_PATH);

        normalizeData(dataSet1);
        normalizeData(dataSet2);

        int numFeatures = dataSet1.get(0).length - 1; // Features (exclude label)
        int numClasses = 10; // Assuming digits 0-9

        NaiveBayes nb = new NaiveBayes(numClasses, numFeatures);

        // Fold 1: Train on dataSet1, test on dataSet2
        nb.train(dataSet1);
        double accuracyFold1 = evaluate(nb, dataSet2);

        // Fold 2: Train on dataSet2, test on dataSet1
        nb.train(dataSet2);
        double accuracyFold2 = evaluate(nb, dataSet1);
        System.out.println("Accuracy Fold 1: " + String.format("%.2f", accuracyFold1 * PERCENT_CONVERSION) + "%");
        System.out.println("Accuracy Fold 2: " + String.format("%.2f", accuracyFold2 * PERCENT_CONVERSION) + "%");
        // Average accuracy
        double averageAccuracy = (accuracyFold1 + accuracyFold2) / 2;
        System.out.println("Average Accuracy: " + String.format("%.2f", averageAccuracy * PERCENT_CONVERSION) + "%");
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

    private static double evaluate(NaiveBayes nb, ArrayList<double[]> testData) {
        int correct = 0;

        for (double[] instance : testData) {
            double[] features = Arrays.copyOf(instance, instance.length - 1);
            int actualLabel = (int) instance[instance.length - 1];

            int prediction = nb.predict(features);
            if (prediction == actualLabel) {
                correct++;
            }
        }

        return (double) correct / testData.size();
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
}
