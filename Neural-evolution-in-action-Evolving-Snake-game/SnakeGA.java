/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snake;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.netgene.ga.GenerationTracker;
import org.netgene.ga.GeneticAlgorithm;
import org.netgene.ga.GeneticConfiguration;
import org.netgene.ga.chromosome.DoubleChromosome;
import org.netgene.ga.core.Individual;
import org.netgene.ga.core.Population;
import org.netgene.ga.gene.DoubleGene;
import org.netgene.ga.selection.parent.TournamentSelector;
import org.netgene.network.MultiLayerNetwork;

/**
 *
 * @author Benjamin Miller
 */
public class SnakeGA 
{
    private static MultiLayerNetwork[] multiLayerNetworks;
    
    private static int populationSize = 1000;
    
    public static void main(String[] args) throws Exception
    {
        multiLayerNetworks = new MultiLayerNetwork[populationSize];
        generateNetworks();  
        
        Population population = new Population();
        
        for(int i=0; i<populationSize; i++)
        {
            DoubleChromosome chromosome = new DoubleChromosome();
            Double weights[] = multiLayerNetworks[i].getNetworkWeights();
            for(int j=0; j<weights.length; j++)
            {
                DoubleGene gene = new DoubleGene(weights[j]);
                chromosome.addGene(gene);
            }
            Individual individual = new Individual(chromosome);
            population.addIndividual(individual);
        }
        
        TournamentSelector ts = new TournamentSelector(10);
        GeneticAlgorithm ga = new GeneticConfiguration()
                                                  .setParentSelector(ts)
                                                  .setElitismSize(10)
                                                  .setMaxGeneration(1000)
                                                  .getAlgorithm();
        
        GenerationTracker debugStep = (g, r) ->
        {
            System.out.println("----------------------------------------------------");
            System.out.println("Generation: " + g.getGeneration());
            System.out.println("Evolution execution: " + r.getEvolutionDuration().toMillis() + "ms");
            System.out.println("Evaluation execution: " + r.getEvaluationDuration().toMillis() + "ms");
            System.out.println("Generation execution: " + r.getGenerationDuration().toMillis()+ "ms");
            System.out.println("Best fitness score: " +g.getPopulation().getBestIndividual().getFitnessScore());
            System.out.println("Longest snake size: " + g.getPopulation().getBestIndividual().getCustomData());
            DoubleChromosome chromosome = (DoubleChromosome) g.getPopulation().getBestIndividual().getChromosome();
            double weights[] = chromosome.toArray();
            MultiLayerNetwork bestNetwork = generateBrain();
            bestNetwork.setNetworkWeights(weights);
            try {
                if(r.getGenerationNumber() == 10)
                    bestNetwork.saveNetwork("mySnake_10.txt");
                if(r.getGenerationNumber() == 100)
                    bestNetwork.saveNetwork("mySnake_100.txt");
                if(r.getGenerationNumber() == 400)
                    bestNetwork.saveNetwork("mySnake_400.txt");
            } 
            catch (IOException ex) 
            {
                System.out.println("IOException! " + ex.toString());
            }
        };
        
        SnakeFitness snakeFitness = new SnakeFitness();
        ga.setGenerationTracker(debugStep);
        ga.evolve(population, snakeFitness);
        
        
        System.out.println("------------------------------------------");
        Individual individual = ga.getPopulation().getBestIndividual();
        
        System.out.println("Best Individual fitness: " + individual.getFitnessScore());
        System.out.println("Best size: " + (Integer)individual.getCustomData());
        
        DoubleChromosome chromosome = (DoubleChromosome) individual.getChromosome();
        double weights[] = chromosome.toArray();
        MultiLayerNetwork bestNetwork = generateBrain();
        bestNetwork.setNetworkWeights(weights);
        bestNetwork.saveNetwork("mySnake.txt");
    }
    
    public static MultiLayerNetwork generateBrain() 
    {
        
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork()
                                         .addLayer(24)
                                         .addLayer(15)
                                         .addLayer(4)
                                         .addBiasNeurons()
                                         .build();
        
        return multiLayerNetwork; 
    }
    
    public static void generateNetworks() throws Exception
    {
        MultiLayerNetwork multiLayerNetwork;
        
        for(int i=0; i<populationSize; i++)
        {
            multiLayerNetwork = generateBrain();
            multiLayerNetwork.loadNetwork("mySnake.txt");
            multiLayerNetworks[i] = multiLayerNetwork;           
        }
        
    }
    
}
