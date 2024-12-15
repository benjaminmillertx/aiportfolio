python
#!/usr/bin/env python3
# -- coding: utf-8 --
# PROGRAMMER: Benjamin Miller
# DATE CREATED: October 4, 2023
# REVISED DATE: 
# PURPOSE: This script classifies pet images using a pretrained CNN model, 
# compares the classifications to the actual identities of the pets in the images, 
# and summarizes the performance of the CNN on the classification task. 
# The true identity of the pet is indicated by the filename, so the program 
# must extract the pet label from the filename before classifying the images. 
# This program will evaluate the performance of three different CNN architectures 
# to determine which one provides the best classification.

# Use argparse for expected command line input:
# python check_images.py --dir  --arch  --dogfile 
# Example call:
# python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Import necessary Python modules
from time import time, sleep

# Import functions for lab checks
from print_functions_for_lab_checks import 

# Import custom functions for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main function of the program
def main():
    # TODO 0: Start measuring the total runtime of the program
    start_time = time()

    # TODO 1: Retrieve command line arguments using the get_input_args function
    in_arg = get_input_args()

    # Validate command line arguments
    check_command_line_arguments(in_arg)

    # TODO 2: Get pet labels from the specified directory
    results = get_pet_labels(in_arg.dir)

    # Verify the creation of pet image labels
    check_creating_pet_image_labels(results)

    # TODO 3: Classify images using the specified architecture
    classify_images(in_arg.dir, results, in_arg.arch)

    # Check the results of the classification
    check_classifying_images(results)

    # TODO 4: Adjust results to determine if images are classified as dogs
    adjust_results4_isadog(results, in_arg.dogfile)

    # Verify the adjustment of results for dog classifications
    check_classifying_labels_as_dogs(results)

    # TODO 5: Calculate statistics from the results
    results_stats = calculates_results_stats(results)

    # Check the results statistics
    check_calculating_results(results, results_stats)

    # TODO 6: Print the results of the classification
    print_results(results, results_stats, in_arg.arch, True, True)

    # TODO 0: End measuring the total runtime of the program
    end_time = time()

    # TODO 0: Calculate and print the total runtime in hh:mm:ss format
    total_time = end_time - start_time
    print("\n Total Elapsed Runtime:", 
          str(int((total_time / 3600))) + ":" + 
          str(int((total_time % 3600) / 60)) + ":" + 
          str(int(total_time % 60)))

# Entry point to run the program
if name == "main":
    main()




