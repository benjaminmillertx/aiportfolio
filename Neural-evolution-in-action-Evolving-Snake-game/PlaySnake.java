/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snake;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author CBaba
 */
public class PlaySnake 
{
    static WorldGUI snakeGui = new WorldGUI(20);
    public static void main(String[] args) throws Exception
    {
        snakeGui.create();
        
        
         
        //CheckAction checkAction = new CheckAction();
        
        snakeGui.addHumanController();
        
        //snakeGui.move(0);
        
    }
    
}