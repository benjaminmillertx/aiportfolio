#
# DEVELOPER: Benjamin Miller
# DATE OF CREATION: October 4, 2023
#
# OBJECTIVE: Implement the function get_pet_labels that generates pet labels
# from the filenames of images. This function takes:
# - The directory containing images as image_dir within get_pet_labels function
# and as in_arg.dir for the function call in the main function.
# This function constructs and returns a dictionary named results_dic
# within get_pet_labels function and as results within main.
# The results_dic dictionary has a 'key' that corresponds to the image filename and
# a 'value' that is a list. This list will include the following item
# at index 0: pet image label (string).
#
# ##

# Import necessary modules
from os import listdir

# TODO 2: Define the get_pet_labels function below. Ensure to replace None
# in the return statement with the results_dic dictionary created in this function.
def get_pet_labels(image_dir):
    """
    Generates a dictionary of pet labels (results_dic) based on the filenames of the image files.
    These labels are used to verify the accuracy of the labels returned by the classifier function,
    as the filenames contain the true identity of the pet in the image. Ensure that the pet labels
    are formatted in all lowercase letters and that leading and trailing whitespace characters are removed.
    (e.g., filename = 'Boston_terrier_02259.jpg' results in Pet label = 'boston terrier')
    
    Parameters:
    image_dir - The complete path to the folder containing images to be classified (string)
    
    Returns:
    results_dic - Dictionary with 'key' as image filename and 'value' as a List.
    The list contains the following item: index 0 = pet image label (string)
    """
    
    # Create a list of files in the specified directory
    file_list = listdir(image_dir)
    
    # Initialize lists to hold filenames and corresponding pet labels
    file_names = []
    pet_labels = []
    
    for index in range(len(file_list)):
        # Convert filename to lowercase
        lower_case_image = file_list[index].lower()
        
        # Split the lowercase string by underscores to separate words
        words_in_image = lower_case_image.split("_")
        
        # Initialize an empty string for the pet name
        pet_name = ""
        
        # Loop through the words to check if they are alphabetic
        for word in words_in_image:
            if word.isalpha():
                pet_name += word + " "
        
        # Remove any leading or trailing whitespace
        pet_name = pet_name.strip()
        
        # Append the pet name and filename to their respective lists
        pet_labels.append(pet_name)
        file_names.append(file_list[index])
    
    # Create an empty dictionary to store results (pet labels, etc.)
    results_dic = {}
    
    # Iterate through each file in the directory to extract pet image labels
    for index in range(len(file_list)):
        # Skip files that start with a dot (e.g., .DS_Store) as they are not pet images
        if file_list[index][0] != ".":
            # If the filename is not already in the dictionary, add it along with its pet label
            if file_names[index] not in results_dic:
                results_dic[file_names[index]] = [pet_labels[index]]
            else:
                print("* Warning: Key=", file_names[index], "already exists in results_dic with value =", results_dic[file_names[index]])
    
    return results_dic