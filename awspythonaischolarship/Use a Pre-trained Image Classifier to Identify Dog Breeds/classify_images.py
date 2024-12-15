
#!/usr/bin/env python3
# -- coding: utf-8 --
# 
#
# PROGRAMMER: Benjamin Miller
# DATE CREATED: October 4, 2023
# 
# PURPOSE: Develop a function named classify_images that utilizes the classifier function
# to generate classifier labels and subsequently compares these labels with the pet image labels.
# This function takes the following inputs:
# - The directory containing images, referred to as image_dir within classify_images and as in_arg.dir for the function call in main.
# - The results dictionary, designated as results_dic within classify_images and results for the function call in main.
# - The architecture of the CNN model, specified as model within classify_images and in_arg.arch for the function call in main.
# This function employs the extend method to append items to the list that serves as the 'value' of the results dictionary.
# You will be adding the classifier label at index 1 of the list and the comparison result of the pet and classifier labels at index 2 of the list.
#
# ## 
# Importing the classifier function to classify images using CNN
from classifier import classifier

# TODO 3: Define the classify_images function below, specifically replace the None
# below with the function definition of classify_images. 
# Note that this function does not return anything since the results_dic dictionary is a mutable data type, so no return is necessary.

def classify_images(images_dir, results_dic, model):
    """
    Generates classifier labels using the classifier function, compares pet labels with classifier labels,
    and appends the classifier label and the comparison result to the results dictionary using the extend method.
    Ensure that the classifier labels are formatted to match the pet image labels, which includes converting
    the classifier labels to lowercase and removing any leading or trailing whitespace. For instance, if the
    classifier function returns 'Maltese dog, Maltese terrier, Maltese', the classifier label should be
    'maltese dog, maltese terrier, maltese'. Note that dog names from the classifier function may be a
    string of names separated by commas when a breed has multiple associated names. For example, a pet image
    labeled 'dalmatian' (pet label) will match the classifier label 'dalmatian, coach dog, carriage dog'
    if the classifier function accurately classifies the pet images of dalmatians. 
    PLEASE NOTE: This function utilizes the classifier() function defined in classifier.py. 
    Refer to test_classifier.py for proper usage of the classifier() function to classify images within this function.

    Parameters:
    images_dir - The complete path to the folder containing images to be classified by the classifier function (string)
    results_dic - A dictionary with 'key' as the image filename and 'value' as a List. The list will contain:
        index 0 = pet image label (string) 
        --- where index 1 & index 2 are added by this function ---
        NEW - index 1 = classifier label (string) 
        NEW - index 2 = 1/0 (int) where 1 = match between pet image and classifier labels and 0 = no match between labels
    model - Specifies which CNN model architecture will be employed by the classifier function to classify the pet images; valid values are: resnet, alexnet, vgg (string)

    Returns: None - results_dic is a mutable data type, so no return is needed.
    """
    
    # Iterate through all entries in the results_dic
    for key in results_dic:
        # Classify the images using the classifier function
        model_label = classifier(images_dir + key, model)
        
        # Format the results for comparison with pet image labels
        model_label = model_label.lower().strip()
        
        # Retrieve the pet image label for comparison
        truth = results_dic[key][0]
        
        # Check if the pet image label is present in the classifier label
        if truth in model_label:
            # Append the classifier label and a match indicator (1) to the results dictionary
            results_dic[key].extend((model_label, 1))
        else:
            # Append the classifier label and a non-match indicator (0) to the results dictionary
            results_dic[key].extend((model_label, 0))