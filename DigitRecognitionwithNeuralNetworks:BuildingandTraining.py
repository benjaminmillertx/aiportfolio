import org.netgene.network.MultiLayerNetwork;
import java.text.DecimalFormat;
import org.netgene.network.learning.optimizer.SGD;
import org.netgene.network.training.data.TrainingData;

/**
 * Digit recognition with a 7-segment pattern using a neural network
 * Trained with backpropagation and output printed in human-readable format.
 *
 * @author Benjamin Hunter Miller
 */
public class Lesson5
{
    private static MultiLayerNetwork network;

    public static void main(String[] args) throws Exception
    {
        network = new MultiLayerNetwork()
                    .addLayer(7)    // Input layer: 7 segments
                    .addLayer(12)   // Hidden layer: more neurons for better generalization
                    .addLayer(10)   // Output layer: digits 0-9
                    .addBiasNeurons()
                    .build();

        TrainingData trainingData = new TrainingData();
        addDigitTrainingData(trainingData);

        SGD optimizer = new SGD();
        optimizer.setLearningRate(0.1);          // Recommended for faster convergence
        optimizer.setMaxIterations(100000);
        optimizer.setErrorStopCondition(0.001);  // Stop when error is sufficiently low

        System.out.println("[*] Training started...");
        optimizer.learn(network, trainingData);
        System.out.println("[✓] Training completed.");

        optimizer.printResults();

        printPredictions(trainingData);
        network.saveNetwork("digit-net.txt");
    }

    // Adds digit data rows (0-9) based on 7-segment display
    private static void addDigitTrainingData(TrainingData data)
    {
        data.addDataRow(new Double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0}, oneHot(0));
        data.addDataRow(new Double[]{0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, oneHot(1));
        data.addDataRow(new Double[]{1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0}, oneHot(2));
        data.addDataRow(new Double[]{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0}, oneHot(3));
        data.addDataRow(new Double[]{0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}, oneHot(4));
        data.addDataRow(new Double[]{1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, oneHot(5));
        data.addDataRow(new Double[]{1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0}, oneHot(6));
        data.addDataRow(new Double[]{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, oneHot(7));
        data.addDataRow(new Double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, oneHot(8));
        data.addDataRow(new Double[]{1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0}, oneHot(9));
    }

    // Creates one-hot output array for expected digit
    private static Double[] oneHot(int index)
    {
        Double[] output = new Double[10];
        for (int i = 0; i < 10; i++)
            output[i] = (i == index) ? 1.0 : 0.0;
        return output;
    }

    // Prints prediction results with confidence
    private static void printPredictions(TrainingData data)
    {
        int correct = 0;

        for (int i = 0; i < data.size(); i++)
        {
            Double[] input = data.getInputData(i);
            Double[] result = network.getNetworkOutput(input);
            int predicted = maxIndex(result);
            int actual = maxIndex(data.getOutputData(i));

            System.out.println("------------------------------------------------------------");
            System.out.println("Expected: " + actual + " | Predicted: " + predicted + " --> "
                               + (predicted == actual ? "✓" : "✗"));
            humanReadable(result);

            if (predicted == actual)
                correct++;
        }

        System.out.printf("\n[✓] Accuracy: %.2f%% (%d/10 correct)\n", (correct / 10.0) * 100, correct);
    }

    // Returns index of max value in array
    private static int maxIndex(Double[] values)
    {
        double max = values[0];
        int index = 0;
        for (int i = 1; i < values.length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                index = i;
            }
        }
        return index;
    }

    // Prints each digit output with confidence
    private static void humanReadable(Double[] result)
    {
        Double max = findMax(result);
        for (int i = 0; i < result.length; i++)
        {
            Double value = Double.parseDouble(new DecimalFormat("##.##").format(result[i]));
            System.out.print("Digit " + i + ": " + (value * 100) + "%");
            if (result[i].equals(max))
                System.out.println(" ***");
            else
                System.out.println();
        }
    }

    // Finds the maximum value in the output array
    public static Double findMax(Double[] result)
    {
        Double max = result[0];
        for (int i = 1; i < result.length; i++)
        {
            if (max < result[i])
                max = result[i];
        }
        return max;
    }
}

 Output Sample (After Training)

------------------------------------------------------------
Expected: 0 | Predicted: 0 --> ✓
Digit 0: 99.0% ***
Digit 1: 0.1%
...
Accuracy: 100.00% (10/10 correct)
