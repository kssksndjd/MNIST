package CW_1;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;

public class AugmentData {
    private static final Random random = new Random();

    // Constant for noise standard deviation
    private static final double NOISE_STD_DEV = 0.02;


    /**
     * Augments the dataset by adding Gaussian noise to features.
     *
     * @param originalData   Original dataset to augment
     * @param augmentFactor  Number of augmented instances per original instance
     * @return Augmented dataset
     */
    public static ArrayList<double[]> augmentData(ArrayList<double[]> originalData, int augmentFactor) {
        ArrayList<double[]> augmentedData = new ArrayList<>(originalData);

        for (double[] instance : originalData) {
            for (int augmentationIndex = 0; augmentationIndex < augmentFactor; augmentationIndex++) {
                double[] augmentedInstance = createAugmentedInstance(instance);
                augmentedData.add(augmentedInstance);
            }
        }

        return augmentedData;
    }

    /**
     * Creates an augmented instance by adding Gaussian noise to the features.
     *
     * @param originalInstance Original data instance
     * @return Augmented instance with noise
     */
    private static double[] createAugmentedInstance(double[] originalInstance) {
        double[] augmentedInstance = Arrays.copyOf(originalInstance, originalInstance.length);

        for (int featureIndex = 0; featureIndex < augmentedInstance.length - 1; featureIndex++) { // Exclude label
            augmentedInstance[featureIndex] += random.nextGaussian() * NOISE_STD_DEV;
        }

        return augmentedInstance;
    }
}
