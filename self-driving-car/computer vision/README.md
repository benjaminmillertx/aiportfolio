# Computer Vision Section
This section an algorithm for following the road will be created.

# TODO:

- Run the file "1_image_threshold.py", I recommend to use the Python debugger to understand how the thresholding operation works
- Open Webots by double clicking the file "city.wbt"
- Run the code the file "_camera_pid.py" as it is to ensure that everything works and make sure that you understand the logic behind it
- Go to the file "_computer_vision/utils.py"
- Remove the content of function "calculate_normalized_average_col" by leaving it as:
```
def calculate_normalized_average_col(binary_image):
    # TODO: implementation it
    return normalized_column, int(average_column)
```
- Implement the function
    - The input is the binary image as a np.array
    - Goal: this function main objective is to return the center of mass of the image (roughly the middle of the middle of the yellow points) normalized (0 is center, negative left part of the image, positive right part of the image)
    - Note: the 2nd term "average_column" is optional, if you don't want to calculate it simply comment out the following line in the "2_camera_pid.py" file (but you won't have a display)
    `# utils.display_binary_image(display_th, binary_image, average_column)`
