// Package declaration to organize code related to the snake game demo.
package snake;

// Import statements for logging and neural network functionality.
import java.util.logging.Level;
import java.util.logging.Logger;
import org.netgene.network.MultiLayerNetwork; // Assumed custom library for neural network functionality.
import org.netgene.network.exception.NNException; // Custom exception handling for the neural network.

/**
 * TestSnake class to demonstrate the integration of a neural network with a snake game.
 */
public class TestSnake {
    /**
     * Main method to run the snake game with neural network-based decision-making.
     */
    public static void main(String[] args) throws Exception {
        // Initialize the neural network with a specific architecture.
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork()
                                         .addLayer(24) // Input layer with 24 neurons.
                                         .addLayer(15) // Hidden layer with 15 neurons.
                                         .addLayer(4)  // Output layer with 4 neurons, representing direction.
                                         .addBiasNeurons() // Adds bias neurons to each layer except the output.
                                         .build(); // Finalizes the network construction.
        
        // Loads a pre-trained neural network model from a file.
        multiLayerNetwork.loadNetwork("mySnake.txt");
        
        // Initialize the game world with a grid size of 10x10.
        WorldGUI world = new WorldGUI(10);
        // Optional: Set the visual size of each square in the grid (commented out).
        // world.setSquaresSize(10);
        // Create the game world and necessary components.
        world.create();
        
        // Define arrays for neural network inputs and outputs.
        Double inputs[];
        int snakeSize = 0; // Track the size of the snake.
        Double outputs[];
        
        // Game loop: continue until the snake is dead or wins by filling the grid.
        while (!world.isDead() && !world.isWinner()) {
            inputs = world.getVision(); // Get current vision inputs for the neural network.
            snakeSize = world.getSnakeSize(); // Update the snake size.
            outputs = multiLayerNetwork.getNetworkOutput(inputs); // Get the neural network's output.
                
            // Determine the direction with the highest output value.
            int direction = 0;
            double max = 0;
            for (int j = 0; j < outputs.length; j++) {
                if (outputs[j] > max) {
                    max = outputs[j];
                    direction = j; // Update direction based on the highest output.
                }
            }
            
            // Move the snake in the determined direction and sleep to slow down the game loop.
            world.move(direction);
            sleep(10); // Pause for a short duration for a better visual experience.
        }
        
        // After the game loop ends, print the final size of the snake.
        System.out.println("Snake size: " + world.getSnakeSize());
    }
    
    /**
     * Utility method to pause the execution thread for a specified duration.
     * @param ms The duration of the pause in milliseconds.
     */
    private static void sleep(int ms) {
        try {
            Thread.sleep(ms); // Attempt to sleep for the specified duration.
        } catch (InterruptedException ex) {
            // Log the interrupted exception if the sleep is interrupted.
            Logger.getLogger(WorldGUI.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
