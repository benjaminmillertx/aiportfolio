#!/usr/bin/env python3
# -- coding: utf-8 --
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Benjamin Hunter Miller
# DATE CREATED: October 4, 2023
#
# PURPOSE: Develop a function that collects the following three command line inputs
# from the user utilizing the Argparse Python module. If the user neglects to
# supply some or all of the three inputs, then the default values are
# applied for the missing inputs. Command Line Arguments:
# 1. Image Directory as --dir with default value 'pet_images'
# 2. CNN Model Type as --arch with default value 'vgg'
# 3. Text File containing Dog Names as --dogfile with default value 'dognames.txt'
# ##
# Import necessary python modules
import argparse

# TODO 1: Define the get_input_args function below; ensure to replace None
# in the return statement with parser.parse_args() parsed argument
# collection that you created with this function
def get_input_args():
    """
    Collects and processes the three command line arguments provided by the user when they execute the program from a terminal window. This function employs Python's argparse module to create and define these three command line arguments. If the user neglects to provide some or all of the three arguments, then the default values are utilized for the missing arguments. Command Line Arguments: 
    1. Image Directory as --dir with default value 'pet_images' 
    2. CNN Model Type as --arch with default value 'vgg' 
    3. Text File containing Dog Names as --dogfile with default value 'dognames.txt' 
    This function returns these arguments as an ArgumentParser object. 
    Parameters: None - simply using argparse module to create & store command line arguments 
    Returns: parse_args() - data structure that stores the command line arguments object
    """
    # Initialize parser using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create three command line arguments as specified above using add_argument() from ArgumentParser method
    parser.add_argument('--dir', type=str, default='pet_images/', help='path to the directory of pet images')
    parser.add_argument('--arch', type=str, default='vgg', help='path to the model classifier')
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help='path to the text file with dog names')
    
    # ---------------------------------------------
    # Note:
    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    
    # Accesses values of Argument 1 by printing it
    print("Argument 1:", in_args.dir)
    # ----------------------------------------------
    
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()

