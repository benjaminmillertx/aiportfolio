/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snake;
import org.netgene.ga.core.Individual;
import org.netgene.ga.chromosome.DoubleChromosome;
import org.netgene.ga.fitness.FitnessFunction;
import org.netgene.network.MultiLayerNetwork;

/**
 *
 * @author Benjamin Miller
 */
public class SnakeFitness implements FitnessFunction
{

    @Override
    public void calculateFitness(Individual indvdl) {
        DoubleChromosome chromosome = (DoubleChromosome) indvdl.getChromosome();
        double weights[] = chromosome.toArray();
        MultiLayerNetwork multiLayerNetwork = SnakeGA.generateBrain();
        multiLayerNetwork.setNetworkWeights(weights);
        Double inputs[] = new Double[24];
        int snakeSize = 0;
        int lifetime = 0;
        int leftToLive = 20;
        int step = 30;
        
        World world = new World(10);
        while(!world.isDead() && !world.isWinner())
        {
            inputs = world.getVision();
            Double outputs[] = multiLayerNetwork.getNetworkOutput(inputs);
            int direction = 0;
            double max = 0;
            for(int i=0; i<outputs.length; i++)
            {
                if(outputs[i] > max)
                {
                    max = outputs[i];
                    direction = i;
                }
            }
            snakeSize = world.getSnakeSize();
            world.move(direction);
            lifetime++;
            
            if(snakeSize == world.getSnakeSize())
            {
                leftToLive--;
            }
            else
            {
                leftToLive = leftToLive + step;
            }
            if(leftToLive == 0)
            {
                break;
            }
        }
        
        double fitness = 0;
        
        if(world.getSnakeSize() <= 9)
        {
            fitness = lifetime * Math.pow(2,world.getSnakeSize());
        }
        else 
        {
            fitness = lifetime * Math.pow(2, 10) * world.getSnakeSize();
        }
        
        indvdl.setFitnessScore(fitness);
        indvdl.setCustomData(world.getSnakeSize());
    }
    
}
