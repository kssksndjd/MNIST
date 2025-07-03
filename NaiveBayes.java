package cw;

import java.util.*;

public class NaiveBayes {

    private final int numClasses;  // Number of classes
    private final int numFeatures; // Number of features

    private double[] classPriors; // Prior probabilities for each class
    private double[][][] featureLikelihoods; // Gaussian parameters for each class and feature

    /**
     * Constructor to initialize Naive Bayes with the dataset dimensions.
     *
     * @param numClasses  Number of output classes
     * @param numFeatures Number of input features
     */
    public NaiveBayes(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.classPriors = new double[numClasses];
        this.featureLikelihoods = new double[numClasses][numFeatures][2]; // [Class][Feature][Mean/Variance]
    }

    /**
     * Trains the Naive Bayes model on the given dataset.
     *
     * @param trainingData Training dataset
     */
    public void train(ArrayList<double[]> trainingData) {
        int[] classCounts = new int[numClasses];
        double[][] featureSums = new double[numClasses][numFeatures];
        double[][] featureSquaredSums = new double[numClasses][numFeatures];

        // Compute class counts and feature sums
        for (double[] instance : trainingData) {
            int actualClass = (int) instance[instance.length - 1]; // Extract class label
            classCounts[actualClass]++;

            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                featureSums[actualClass][featureIndex] += instance[featureIndex];
                featureSquaredSums[actualClass][featureIndex] += Math.pow(instance[featureIndex], 2);
            }
        }

        // Estimate priors and Gaussian parameters
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            classPriors[classIndex] = (double) classCounts[classIndex] / trainingData.size();

            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                double mean = featureSums[classIndex][featureIndex] / classCounts[classIndex];
                double variance = (featureSquaredSums[classIndex][featureIndex] / classCounts[classIndex]) - Math.pow(mean, 2);
                featureLikelihoods[classIndex][featureIndex][0] = mean;
                featureLikelihoods[classIndex][featureIndex][1] = variance > 0 ? variance : 1e-6; // Avoid division by zero
            }
        }
    }

    /**
     * Predicts the class label for a given input instance.
     *
     * @param features Input features
     * @return Predicted class label
     */
    public int predict(double[] features) {
        double[] logPosteriors = new double[numClasses];

        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            logPosteriors[classIndex] = Math.log(classPriors[classIndex]);

            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                double mean = featureLikelihoods[classIndex][featureIndex][0];
                double variance = featureLikelihoods[classIndex][featureIndex][1];
                double logLikelihood = -0.5 * Math.log(2 * Math.PI * variance)
                        - Math.pow(features[featureIndex] - mean, 2) / (2 * variance);

                logPosteriors[classIndex] += logLikelihood;
            }
        }

        // Return the class with the highest posterior probability
        int predictedClass = 0;
        for (int classIndex = 1; classIndex < numClasses; classIndex++) {
            if (logPosteriors[classIndex] > logPosteriors[predictedClass]) {
                predictedClass = classIndex;
            }
        }
        return predictedClass;
    }
}
