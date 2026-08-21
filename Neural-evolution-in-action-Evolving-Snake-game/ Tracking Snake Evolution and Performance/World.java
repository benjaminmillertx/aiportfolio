package snake; // Package declaration, grouping related classes together in a namespace.

import java.util.ArrayList; // Importing the ArrayList class for dynamic arrays.
import java.util.Random; // Importing the Random class to generate random numbers.

/**
 * The World class represents the game environment for the snake game.
 * It manages the grid, the snake, food placement, and game state.
 */
public class World {
    private ArrayList<Node> nodes = new ArrayList(); // List of all nodes (grid cells) in the game world.
    public ArrayList<Node> snake = new ArrayList(); // List of nodes that make up the snake's body.
    
    public int snakeHead; // Index of the node representing the snake's head.
    
    private int worldSize; // The size of the game world (width and height of the square grid).
    
    private int snakeSize = 3; // Initial size of the snake (not actively used in this version).
    
    public int movingDirection = 0; // Current moving direction of the snake.
    
    // Direction constants to clarify the code when changing the snake's direction.
    private final int MOVE_UP = 0;
    private final int MOVE_LEFT = 1;
    private final int MOVE_RIGHT = 2;
    private final int MOVE_DOWN = 3;
    
    // Vision directions for calculating distances to walls, the snake's body, and food.
    private final int LOOK_UP = 0;
    private final int LOOK_LEFT = 1;
    private final int LOOK_RIGHT = 2;
    private final int LOOK_DOWN = 3;
    private final int LOOK_UP_LEFT = 4;
    private final int LOOK_UP_RIGHT = 5;
    private final int LOOK_DOWN_LEFT = 6;
    private final int LOOK_DOWN_RIGHT = 7;
    
    public boolean isEating = false; // Flag indicating whether the snake has just eaten food.
    
    private boolean isDead = false; // Flag indicating whether the snake is dead.
    
    private boolean hitTheWall = false; // Flag for wall collision (unused in this version).
    
    private final int initialSize = 3; // Initial length of the snake.
    
    private int foodNodeNr; // Index of the node containing food.
    
    private int move = 0; // Counter for moves made (unused in this version).

    /**
     * Constructor for the World class. Initializes the game world.
     * @param worldSize The size of the world. Throws an exception if smaller than 7.
     * @throws Exception if the world size is smaller than 7.
     */
    public World(int worldSize) throws RuntimeException {
        if (worldSize < 7)
            throw new RuntimeException("World size cannot be smaller than 7");
        this.worldSize = worldSize;
        // Initialize the nodes of the game world.
        for (int i = 0; i < worldSize * worldSize; i++) {
            Node node = new Node(i);
            nodes.add(node);
        }
        
        createSnake(); // Initialize the snake in the game world.
        createFood(); // Place the first piece of food in the game world.
    }

    /**
     * Initializes the snake at a specific position and direction.
     */
    private void createSnake() {
        snake.clear(); // Clear any existing snake from the game world.
        // Calculate the initial position of the snake's head.
        snakeHead = worldSize * (2 * worldSize / 3) + (2 * worldSize / 3);
        
        // Add nodes to the snake ArrayList to represent the snake's body.
        for (int i = 0; i < initialSize; i++) {
            snake.add(nodes.get(snakeHead + i * worldSize));
        }
        movingDirection = MOVE_UP; // Set the initial movement direction of the snake.
        isDead = false; // Initially, the snake is not dead.
    }

    /*
     1 - UP
     2 - LEFT
     3 - RIGHT
     4 - DOWN
    */
    /**
     * Moves the snake in the specified direction.
     * @param direction The direction in which to move the snake.
     */
    public void move(int direction) {
        move++; // Increment the move counter.
        // Check if the new direction is directly opposite to the current direction, which is not allowed.
        if(direction == MOVE_UP && movingDirection == MOVE_DOWN)
        {
            isDead = true;
            return; //do nothing, cannot change the moving direction
        }
        if(direction == MOVE_DOWN && movingDirection == MOVE_UP)
        {
            isDead = true;
            return; //do nothing, cannot change the moving direction
        }
        if(direction == MOVE_LEFT && movingDirection == MOVE_RIGHT)
        {
            isDead = true;
            return; //do nothing, cannot change the moving direction
        }
        if(direction == MOVE_RIGHT && movingDirection == MOVE_LEFT)
        {
            isDead = true;
            return; //do nothing, cannot change the moving direction
        }

        if (checkCollision(direction)) { // Check for collisions with walls or the snake itself.
            isDead = true; // The snake dies on collision.
            return; // Exit the method.
        }

        // Update the snake's head position based on the direction of movement.
        switch (direction) {
            case MOVE_UP:
                snakeHead = snakeHead - worldSize;
                break;
            case MOVE_LEFT:
                snakeHead = snakeHead - 1;
                break;
            case MOVE_RIGHT:
                snakeHead = snakeHead + 1;
                break;
            case MOVE_DOWN:
                snakeHead = snakeHead + worldSize;
                break;
        }

        if (snake.contains(nodes.get(snakeHead))) { // Check if the new head position is on the snake's body.
            isDead = true; // The snake dies if it collides with itself.
            return; // Exit the method.
        }

        snake.add(0, nodes.get(snakeHead)); // Add the new head to the snake.
        
        if (nodes.get(snakeHead).isFood()) { // Check if the snake's new head position contains food.
            nodes.get(snakeHead).setIsFood(false); // The food is consumed.
            isEating = true; // Set the isEating flag to true.
            if (isWinner())
                return; // Check if consuming this food wins the game.
            createFood(); // Place new food in the game world.
        } else {
            isEating = false; // The snake did not eat food this turn.
        }
        
        if (!isEating) {
            snake.remove(snake.size() - 1); // Remove the tail of the snake, unless it has just eaten food.
        }
        
        movingDirection = direction; // Update the snake's moving direction.
    }

    /**
     * Checks for collisions based on the direction of movement.
     * @param direction The direction in which the snake is moving.
     * @return true if a collision occurs, false otherwise.
     */
    private boolean checkCollision(int direction) {
        // Check if moving in the specified direction causes a collision with the wall.
       //check if snake hits the wall
        if(direction == MOVE_UP && snakeHead < worldSize)
            return true;
        if(direction == MOVE_LEFT && snakeHead % worldSize == 0)
            return true;
        if(direction == MOVE_RIGHT  && (snakeHead % worldSize == worldSize -1))
            return true;
        if(direction == MOVE_DOWN && (snakeHead + worldSize > worldSize * worldSize - 1))
            return true;
    
        return false; // No collision detected.
    }

    /**
     * Places food in a random location in the game world that is not occupied by the snake.
     */
    public void createFood() {
        ArrayList<Node> cloneList = (ArrayList) nodes.clone(); // Clone the list of nodes to find an empty spot.
        
        for(int i = 0; i<snake.size(); i++)
        {
            cloneList.remove(snake.get(i));  // Remove nodes occupied by the snake.
        }
                
        Random random = new Random();
        
        int nextInt = random.nextInt(cloneList.size()); // Pick a random node from the remaining nodes.
        
        Node food = cloneList.get(nextInt); // Get the node to place the food.
        
        foodNodeNr = food.getIndex(); // Store the index of the food node.
 
        food.setIsFood(true); // Mark the node as containing food.
    }

    /**
     * Calculates distances to walls, the snake's body, and food from the snake's head in various directions.
     * @return An array of distances in all directions.
     */
    public Double[] getVision() {
        Double[] vision = new Double[24]; // Array to store vision distances.
        
        // Calculate distances for each direction.
         vision[0] = distanceToWall(this.LOOK_UP);
        //System.out.println("distance wall up: " + vision[0]);
        vision[1] = distanceToBody(this.LOOK_UP);
        vision[2] = distanceToFood(this.LOOK_UP);
        //System.out.println("food wall up: " + vision[2]);
        
        vision[3] = distanceToWall(this.LOOK_LEFT);
        //System.out.println("distance wall left: " + vision[3]);
        vision[4] = distanceToBody(this.LOOK_LEFT);
        vision[5] = distanceToFood(this.LOOK_LEFT);
        //System.out.println("food wall left: " + vision[5]);
          
        vision[6] = distanceToWall(this.LOOK_RIGHT);
        //System.out.println("distance wall right: " + vision[6]);
        vision[7] = distanceToBody(this.LOOK_RIGHT);
        vision[8] = distanceToFood(this.LOOK_RIGHT);
        //System.out.println("food wall right: " + vision[8]);
        
        vision[9] = distanceToWall(this.LOOK_DOWN);
        //System.out.println("distance wall down: " + vision[9]);
        vision[10] = distanceToBody(this.LOOK_DOWN);
        vision[11] = distanceToFood(this.LOOK_DOWN);
        //System.out.println("food wall down: " + vision[11]);
        
        vision[12] = distanceToWall(this.LOOK_UP_LEFT);
        //System.out.println("distance wall up left: " + vision[12]);
        vision[13] = distanceToBody(this.LOOK_UP_LEFT);
        vision[14] = distanceToFood(this.LOOK_UP_LEFT);
        //System.out.println("food wall up left: " + vision[14]);
         
        vision[15] = distanceToWall(this.LOOK_UP_RIGHT);
        //System.out.println("distance wall up right: " + vision[15]);
        vision[16] = distanceToBody(this.LOOK_UP_RIGHT);
        vision[17] = distanceToFood(this.LOOK_UP_RIGHT);
        //System.out.println("food wall up right: " + vision[17]);
        
        vision[18] = distanceToWall(this.LOOK_DOWN_LEFT);
        //System.out.println("distance wall down left: " + vision[18]);
        vision[19] = distanceToBody(this.LOOK_DOWN_LEFT);
        vision[20] = distanceToFood(this.LOOK_DOWN_LEFT);
        //System.out.println("food wall down left: " + vision[20]);
        
        vision[21] = distanceToWall(this.LOOK_DOWN_RIGHT);
        //System.out.println("distance wall down right: " + vision[21]);
        vision[22] = distanceToBody(this.LOOK_DOWN_RIGHT);
        vision[23] = distanceToFood(this.LOOK_DOWN_RIGHT);
        //System.out.println("food wall down right: " + vision[23]);
        return vision; // Return the array of vision distances.
    }

    /**
     * Calculates the distance from the snake's head to the nearest wall in a given direction.
     * @param direction The direction in which to look.
     * @return The normalized distance to the nearest wall.
     */
    public double distanceToWall(int direction) {
        double distance = 0;
        // Calculate distance based on direction.
        switch (direction) {
            case LOOK_UP:
                distance = snakeHead / worldSize;
                break;
            case LOOK_DOWN:
                distance = worldSize - snakeHead / worldSize - 1;
                break;
            case LOOK_LEFT:
                distance = snakeHead % worldSize;
                break;
            case LOOK_RIGHT:
                distance = worldSize - snakeHead % worldSize - 1;
                break;
            case LOOK_UP_LEFT:
                distance = Math.min(snakeHead / worldSize, snakeHead % worldSize);
                break;
            case LOOK_UP_RIGHT:
                distance = Math.min(snakeHead / worldSize, worldSize - snakeHead % worldSize - 1);
                break;
            case LOOK_DOWN_LEFT:
                distance = Math.min(worldSize - snakeHead / worldSize - 1, snakeHead % worldSize);
                break;
            case LOOK_DOWN_RIGHT:
                distance = Math.min(worldSize - snakeHead / worldSize - 1, worldSize - snakeHead % worldSize - 1);
                break;
        }
        
        distance = distance + 1; // Normalize distance.
        
        distance = 1 / distance; // Invert distance for ease of use.
           
        return distance; // Return the calculated distance.
    }

    /**
     * Calculates the distance from the snake's head to its body in a given direction.
     * @param direction The direction in which to look.
     * @return The normalized distance to the snake's body, or 0 if no body part is found.
     */
   public double distanceToBody(int direction)
    {
        double distance = 0;
        boolean maxCount = true;
        int i = snakeHead;
        while(true)
        {
            switch(direction)
            {
                case LOOK_UP:
                    i = i - worldSize;
                    maxCount = (i >= 0);
                break;
                case LOOK_LEFT:
                    i = i - 1;
                    maxCount = (i % worldSize != worldSize - 1) && (i >= 0);  // - 1
                break;
                case LOOK_RIGHT:
                    i = i + 1;
                    maxCount = (i % worldSize != 0) && (i < worldSize * worldSize -1);
                break;
                case LOOK_DOWN:
                    i = i + worldSize;
                    maxCount = i < worldSize * worldSize - 1;
                break;
                case LOOK_UP_LEFT:
                    i = i - worldSize - 1;
                    maxCount = (i >= 0) && (i % worldSize != worldSize - 1);
                break;
                case LOOK_UP_RIGHT:
                    i = i - worldSize + 1;
                    maxCount = (i >= 0) && (i % worldSize != 0) && (i < worldSize * worldSize -1);
                break;
                case LOOK_DOWN_LEFT:
                    i = i + worldSize - 1;
                    maxCount = (i < worldSize * worldSize - 1) && (i % worldSize != worldSize - 1) && (i >= 0);
                break;
                case LOOK_DOWN_RIGHT:
                    i = i + worldSize + 1;
                    maxCount = (i < worldSize * worldSize - 1) && (i % worldSize != 0);
                break;
            }
            
            if(!maxCount)
                break;  // Stop if we reach the edge of the grid.
            
            distance++;
            
            if(snake.contains(nodes.get(i)))
            {
                return 1/distance;    // Return the normalized distance to the body.
            }
        }
        
        return 0;   // Return 0 if no body part is found in the given direction.
    }

    /**
     * Calculates the distance from the snake's head to the food in a given direction.
     * @param direction The direction in which to look.
     * @return The normalized distance to the food, or 0 if food is not found in that direction.
     */
    public double distanceToFood(int direction)
    {   
        int distance = 1;
        int snakeLineX = snakeHead % worldSize;
        int foodLineX = foodNodeNr % worldSize;
        
        int snakeLineY = snakeHead / worldSize ;
        int foodLineY = foodNodeNr / worldSize;
        
        double dx = Math.abs(snakeLineX - foodLineX);
        double dy = Math.abs(snakeLineY - foodLineY);
                
        if(direction == this.LOOK_UP)
        {
            if(dx == 0 && foodLineY < snakeLineY )
                 return distance;
        }
        if(direction == this.LOOK_LEFT)
        {
            if(dy == 0 && foodLineX < snakeLineX)
                return distance ;
        }
        if(direction == this.LOOK_RIGHT)
        {
            if(dy == 0 && foodLineX > snakeLineX)
                return distance ;
        }
        if(direction == this.LOOK_DOWN)
        {
            if(dx ==0 && foodLineY > snakeLineY)
                return distance ;
        }
        
        if(direction == LOOK_UP_LEFT && dx == dy && foodLineX < 0)  
        {
            return distance ;
        }
        if(direction == LOOK_UP_RIGHT && dx == dy && foodLineX > 0)
        {
            return distance ;
        }
        if(direction == LOOK_DOWN_LEFT && dx == dy && foodLineX < 0)
        {
            return distance ;
        }
        if(direction == LOOK_DOWN_RIGHT && dx == dy && foodLineX > 0)
        {
            return distance ;
        }
 
        return 0;
    }
    
    // Getters for various game state properties.
    public int getSnakeHead() {
        return this.snakeHead;
    }
    
    public int getSnakeSize() {
        return snake.size();
    }
    
    public boolean isDead() {
        return isDead;
    }
    
    public int getSnakeTail() {
        return snake.get(snake.size() - 1).getIndex();
    }
  
    public int getMovingDirection() {
        return this.movingDirection;
    }
    
    public int getFoodNodeNr() {
        return this.foodNodeNr;
    }
    
    /**
     * Checks if the snake has won the game by occupying the entire game world.
     * @return true if the snake size equals the total number of nodes, false otherwise.
     */
    public boolean isWinner() {
        return snake.size() == worldSize * worldSize;
    }
  
    /**
     * The Node class represents a single cell in the game world.
     * It can be part of the snake, contain food, or be empty.
     */
    public class Node {
        private boolean isFood = false; // Flag indicating whether this node contains food.
        private boolean isSnake = false; // Flag indicating whether this node is part of the snake (unused in this version).
        
        private int index; // The index of this node in the game world.
       
        public Node(int index) {
            this.index = index; // Constructor sets the index of the node.
        }
        
        // Setters and getters for the node properties.
        public void setIsFood(boolean isFood) {
            this.isFood = isFood;
        }

        public void setIndex(int index) {
            this.index = index;
        }
        
        public boolean isSnake() {
            return isSnake;
        }

        public void setIsSnake(boolean isSnake) {
            this.isSnake = isSnake;
        }
        
        public boolean isFood() {
            return isFood;
        }

        public int getIndex() {
            return index;
        }
        
        public String toString() {
            return "Node: " + getIndex(); // For debugging, returns a string representation of the node.
        }
    }
}
