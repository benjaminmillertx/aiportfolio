import org.netgene.network.MultiLayerNetwork;
import org.netgene.network.exception.NNException;
import org.netgene.network.error.function.MeanSquaredError;
import org.netgene.network.training.data.*;

import org.netgene.ga.*;
import org.netgene.ga.core.*;
import org.netgene.ga.chromosome.*;
import org.netgene.ga.crossover.TwoPointCrossover;
import org.netgene.ga.selection.parent.TournamentSelector;
import org.netgene.ga.fitness.FitnessFunction;
import org.netgene.ga.stop.GenerationTracker;

import java.text.DecimalFormat;

/**
 * Digit Recognition using Neural Networks trained by Genetic Algorithms.
 * By Benjamin Hunter Miller
 */
public class DigitRecognitionGA {

    private static final int populationSize = 1000;
    private static final int maxGenerations = 10000;
    private static final double targetError = 0.2;
    private static MultiLayerNetwork bestNetwork;

    public static void main(String[] args) {
        // Initialize and prepare training data for digits 0-9
        TrainingData trainingData = createTrainingData();

        // Prepare a population of networks
        MultiLayerNetwork[] networks = new MultiLayerNetwork[populationSize];
        for (int i = 0; i < populationSize; i++) {
            networks[i] = buildNetwork();
        }

        // Initialize population with random weights
        Population population = new Population();
        for (int i = 0; i < populationSize; i++) {
            DoubleChromosome chromosome = new DoubleChromosome(networks[0].getNetworkWeights().length);
            population.addIndividual(new Individual(chromosome));
        }

        // Define the genetic algorithm configuration
        GeneticAlgorithm ga = new GeneticConfiguration()
            .setParentSelector(new TournamentSelector(10))
            .setCrossoverOperator(new TwoPointCrossover())
            .setElitismSize(10)
            .setMaxGeneration(maxGenerations)
            .setTargetFitness(1 / targetError)
            .getAlgorithm();

        // Fitness function based on network performance (MSE)
        FitnessFunction fitnessFunction = individual -> {
            DoubleChromosome chromosome = (DoubleChromosome) individual.getChromosome();
            double[] weights = chromosome.toArray();

            MultiLayerNetwork net = buildNetwork();
            net.setNetworkWeights(weights);

            DataSet predicted = new DataSet();
            for (int i = 0; i < trainingData.size(); i++) {
                predicted.addDataRow(net.getNetworkOutput(trainingData.getInputData(i)));
            }

            double error = new MeanSquaredError().calculateError(predicted, trainingData.getTagetDataSet());
            individual.setFitnessScore(1 / error);
            individual.setCustomData(error);
        };

        // Log progress for each generation
        ga.setGenerationTracker((generation, result) -> {
            double error = (double) generation.getPopulation().getBestIndividual().getCustomData();
            System.out.printf("Generation %d | Best Fitness: %.5f | Error: %.5f | Eval Time: %d ms\n",
                generation.getGeneration(),
                generation.getPopulation().getBestIndividual().getFitnessScore(),
                error,
                result.getEvaluationDuration().toMillis()
            );
        });

        // Evolve the population
        ga.evolve(population, fitnessFunction);

        // Get the best performing individual
        Individual best = ga.getPopulation().getBestIndividual();
        DoubleChromosome bestChromosome = (DoubleChromosome) best.getChromosome();

        bestNetwork = buildNetwork();
        bestNetwork.setNetworkWeights(bestChromosome.toArray());

        // Display recognition results
        printResults(trainingData);
    }

    /**
     * Builds and returns a fresh neural network.
     */
    private static MultiLayerNetwork buildNetwork() {
        return new MultiLayerNetwork()
            .addLayer(7)     // Input layer (7 segments)
            .addLayer(5)     // Hidden layer
            .addLayer(10)    // Output layer (digits 0–9)
            .addBiasNeurons()
            .build();
    }

    /**
     * Constructs the digit training dataset using segment activations.
     */
    private static TrainingData createTrainingData() {
        double[][] inputs = {
            {1,1,1,1,1,1,0}, // 0
            {0,1,1,0,0,0,0}, // 1
            {1,1,0,1,1,0,1}, // 2
            {1,1,1,1,0,0,1}, // 3
            {0,1,1,0,0,1,1}, // 4
            {1,0,1,1,0,1,1}, // 5
            {1,0,1,1,1,1,1}, // 6
            {1,1,1,0,0,0,0}, // 7
            {1,1,1,1,1,1,1}, // 8
            {1,1,1,1,0,1,1}  // 9
        };

        double[][] outputs = new double[10][10];
        for (int i = 0; i < 10; i++) {
            outputs[i][i] = 1.0; // One-hot encoding
        }

        TrainingData data = new TrainingData();
        for (int i = 0; i < 10; i++) {
            Double[] input = toDoubleArray(inputs[i]);
            Double[] output = toDoubleArray(outputs[i]);
            data.addDataRow(input, output);
        }
        return data;
    }

    /**
     * Converts a primitive double[] to a boxed Double[].
     */
    private static Double[] toDoubleArray(double[] array) {
        Double[] boxed = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            boxed[i] = array[i];
        }
        return boxed;
    }

    /**
     * Prints the recognition results for each digit.
     */
    private static void printResults(TrainingData data) {
        for (int i = 0; i < data.size(); i++) {
            System.out.println("--------------------------------------------------");
            System.out.println("Expected Digit: " + i);
            Double[] result = bestNetwork.getNetworkOutput(data.getInputData(i));
            printProbabilities(result);
        }
    }

    /**
     * Prints probability for each digit with a highlight on the most likely.
     */
    private static void printProbabilities(Double[] result) {
        Double max = findMax(result);
        DecimalFormat format = new DecimalFormat("##.####");

        for (int i = 0; i < result.length; i++) {
            double value = Double.parseDouble(format.format(result[i]));
            System.out.printf("Digit %d: %.2f%% %s\n", i, value * 100, (result[i].equals(max)) ? "***" : "");
        }
    }

    /**
     * Returns the maximum value from the array.
     */
    private static Double findMax(Double[] result) {
        Double max = result[0];
        for (Double val : result) {
            if (val > max) max = val;
        }
        return max;
    }
}
