
#!/usr/bin/env python3
# -- coding: utf-8 --
# Filename: test_classifier.py
# PROGRAMMER: Benjamin Miller
# DATE CREATED: 10/19/2024
# PURPOSE: To demonstrate the proper usage of the classifier() function defined in classifier.py.
# This function uses a CNN model architecture that has been pretrained on the ImageNet data to classify images.
# The accepted model architectures are: 'resnet', 'alexnet', and 'vgg'.
# Usage: python test_classifier.py -- will run the program from the command line.

# Import the classifier function for using pretrained CNN to classify images
from classifier import classifier

# Define a test image from the pet_images folder
test_image = "pet_images/Collie_03797.jpg"

# Define the model architecture to be used for classification
# NOTE: This function only works for model architectures: 'vgg', 'alexnet', 'resnet'
model = "vgg"

# Demonstrate the usage of the classifier() function
# NOTE: image_classification is a text string that contains mixed case (both lower and upper case letters)
# It can include multiple labels separated by commas for labels with more than one word.
image_classification = classifier(test_image, model)

# Print the result from running the classifier() function
print("\nResults from test_classifier.py\nImage:", test_image, 
      "\nUsing model:", model, 
      "\nWas classified as a:", image_classification)