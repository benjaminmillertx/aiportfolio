/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snake;

// Import statements for using AWT and Swing components, event handling, and collections.
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.border.Border;

/**
 * The WorldGUI class creates a graphical user interface for the snake game, allowing visual gameplay.
 */
public class WorldGUI
{
    private World world;  // Instance of the game logic class.
    
    // Constants representing movement directions, mirrored from the World class for consistency.
    private final int MOVE_UP = 0;
    private final int MOVE_LEFT = 1;
    private final int MOVE_RIGHT = 2;
    private final int MOVE_DOWN = 3;
    
    private final int LOOK_UP = 0;
    private final int LOOK_LEFT = 1;
    private final int LOOK_RIGHT = 2;
    private final int LOOK_DOWN = 3;
    private final int LOOK_UP_LEFT = 4;
    private final int LOOK_UP_RIGHT = 5;
    private final int LOOK_DOWN_LEFT = 6;
    private final int LOOK_DOWN_RIGHT = 7;
    
    //-----------------------GUI-----------------------
    // GUI components and layout managers.
    private final ArrayList<JButton> buttonList = new ArrayList(); // List to store buttons representing cells.
    private final ArrayList<JButton> snake = new ArrayList();
    
    private JFrame frame; // The main window of the application.
    private final GridLayout grid; // Grid layout for arranging buttons in a grid pattern.
    
    private int worldSize;  // Size of the game world (grid dimension).
    private int squareLength = 40; // Pixel dimension of each square (cell) in the grid.
    
    public Color defaultColor; // Default color for resetting button backgrounds.
    
    private Border defaultBorder; // Default border for buttons, used for resetting appearance.
    
    private boolean humanController = false; // Flag to enable human control via keyboard.
    
    private int oldTail = 0;
    private int oldSnakeHead = 0;
    
    private int snakeSize;
    
    public boolean isEating = false;
    
    public boolean isWinner = false;
    
    public JLabel statusLabel; // Label to display the game status (e.g., snake size).
    
    private Border lineBorder; // Border for styling components.
    
    private boolean stop = false; // Flag to control game loop, stops game when true
    
    // Initial location of the game window.
    private int locationX = 100;     
    private int locationY = 100;
    
     /**
     * Constructor for setting up the game GUI with a specified world (grid) size.
     */
    public WorldGUI(int worldSize) 
    {
        this.worldSize = worldSize;
        grid = new GridLayout(this.worldSize, this.worldSize); // Initialize grid layout manager.
    }
    
     /**
     * Creates and shows the GUI, initializes game components.
     */
    public void create() throws Exception
    {
        world = new World(worldSize); // Initialize the game world logic.
        frame = new JFrame("Snake GUI"); // Main game window with title.
        //this.worldSize = worldSize;
 
        frame = new JFrame("Snake GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Close button exits the app.
        //frame.setSize(this.worldSize*squareLength, this.worldSize*squareLength);

        //grid = new GridLayout(this.worldSize, this.worldSize);
        
        lineBorder = BorderFactory.createLineBorder(Color.BLACK, 1); // Black border for styling.
        
        // Setting label appearance.
        statusLabel = new JLabel();
        statusLabel.setText("     Snake Size: " + world.getSnakeSize() + "     ");
        statusLabel.setHorizontalAlignment(SwingConstants.CENTER);
        statusLabel.setVerticalAlignment(SwingConstants.CENTER);
        statusLabel.setBorder(lineBorder);
        statusLabel.setFont(new Font(statusLabel.getFont().getFontName(), Font.PLAIN, 16));
        
        JPanel content = new JPanel();
        BorderLayout borderLayout = new BorderLayout(); // Main content pane with border layout.
        //content.setSize(this.worldSize*squareLength, this.worldSize*squareLength);
        content.setLayout(borderLayout);
        borderLayout.setHgap(10);
        borderLayout.setVgap(10);
        
        JPanel buttonsPanel = new JPanel(); 
        buttonsPanel.setSize(this.worldSize*squareLength, this.worldSize*squareLength);
        //grid.set
        buttonsPanel.setLayout(grid);
        buttonsPanel.setBorder(lineBorder); // Set border for grid panel.
        // Layout setup for content pane.
        content.add(buttonsPanel, BorderLayout.CENTER);  // Grid of buttons in the center.
        content.add(statusLabel, BorderLayout.SOUTH);    // Status label at the bottom.
                
        JButton defaultButton = new JButton();
        defaultBorder = defaultButton.getBorder();
        
        // Create and add buttons for each cell in the grid to the buttonsPanel.
        for(int i=0; i< this.worldSize*this.worldSize; i++)
        {
            JButton button = new JButton(); // No border for cells by default.
            button.setBorder(null);         // Disable buttons to prevent them from being clickable.
            buttonList.add(button);         // Add button to list for later access.
            button.setEnabled(false);
            buttonsPanel.add(button);       // Add button to the panel.
        }
        
        defaultColor = buttonList.get(0).getBackground();  // Store the default background color of buttons.
        
        paintSnake();
        frame.setSize(this.worldSize*squareLength, this.worldSize*squareLength + statusLabel.getSize().width); // Set size to fit grid and label.
        frame.setContentPane(content);  // Set the content pane
        frame.setVisible(true);         // Show the window.
        frame.setLocation(locationX, locationY);  
        frame.setLocation(1030, 200); // Override initial location.
    }
    
     /**
     * Updates the visual representation of the snake and food on the grid.
     */
    private void paintSnake()
    {
        snake.add(buttonList.get(world.snake.get(0).getIndex()));  //add head
        snake.get(0).setBackground(Color.GREEN);  //colour Head
        snake.get(0).setBorder(defaultBorder);   
        for(int i=1; i<world.snake.size(); i++)
        {
            buttonList.get(world.snake.get(i).getIndex()).setBackground(Color.red);
            buttonList.get(world.snake.get(i).getIndex()).setBorder(defaultBorder);
            snake.add(buttonList.get(world.snake.get(i).getIndex()));
        }

        buttonList.get(world.getFoodNodeNr()).setBackground(Color.blue);
             
    }
    
     /**
     * Handles the game logic for moving the snake on the grid.
     */
    public void moveSnake()
    {
        //old head becomes body
        snake.get(0).setBackground(Color.RED);
        snake.get(0).setBorder(defaultBorder);
        
        //add new head
        snake.add(0, buttonList.get(world.getSnakeHead()));
        snake.get(0).setBackground(Color.green);
        snake.get(0).setBorder(defaultBorder);
               
            
        if(!world.isEating)
        {
            JButton removeTail = snake.remove(snake.size()-1);
            removeTail.setBackground(defaultColor);
            removeTail.setBorder(null);
            buttonList.get(world.getFoodNodeNr()).setBackground(Color.blue);
        }
         
    }
    
    /**
     * Moves the snake in a specified direction and updates the game state and GUI.
     */
    public void move(int direction)
    {
        if(!stop)
        {
            oldTail = world.getSnakeTail();
            oldSnakeHead = world.getSnakeHead();
            world.move(direction);
            world.getVision();
            moveSnake();
            statusLabel.setText("     Snake Size: " + world.getSnakeSize() + "     ");
            if(world.isWinner())
            {
                statusLabel.setText("!WINNER! Snake Size: " + world.getSnakeSize());
                stop = true;
                return;
            }
            if(world.isDead())
            {
                statusLabel.setText("!SNAKE is DEAD! Snake Size: " + world.getSnakeSize());
                stop = true;
                return;
            }
        }
        
    }
    
    
    public Double[] getVision()
    {
        return world.getVision();
    }
    
    public int getSnakeSize()
    {
        return world.getSnakeSize();
    }
    
    
    /**
     * Adds keyboard controls for human players to control the snake's movement.
     */
    public void addHumanController()
    {
        humanController = true;
        frame.addKeyListener(new java.awt.event.KeyListener() {
            public void keyTyped(KeyEvent e)
            {
                //do nothing
            }

            @Override
            public void keyPressed(KeyEvent e) {
                //if(!keyPressed)
                //{
                    boolean keyPressed = true;
                    if(e.getKeyCode() == KeyEvent.VK_UP)
                    {
                        move(MOVE_UP);
                    }
                    if(e.getKeyCode() == KeyEvent.VK_DOWN)
                    {
                        move(MOVE_DOWN);
                    }
                    if(e.getKeyCode() == KeyEvent.VK_LEFT)
                    {
                        move(MOVE_LEFT);
                    }
                    if(e.getKeyCode() == KeyEvent.VK_RIGHT)
                    {
                        move(MOVE_RIGHT);
                    }
             }

            @Override
            public void keyReleased(KeyEvent e) {
                //do nothing
            }
        }
        
        );
    }
    
    public boolean isWinner()
    {
        return world.isWinner();
    }
    
    public boolean isDead()
    {
        return world.isDead();
    }
    
    public void distroy()
    {
        frame.setVisible(false);
        frame.dispose();
        frame = null;
    }
        
    public void setSleep(int sleep)
    {
        
    }
    
    private void sleep(int ms)
    {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ex) {
            Logger.getLogger(WorldGUI.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}


