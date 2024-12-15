# PROGRAMMER: Benjamin Miller
# DATE CREATED: 10/19/2024
# 
# PURPOSE: This function calculates the statistics of the image classification
# results by processing a dictionary of results from the classifier.
# It returns a dictionary (results_stats_dic) that contains counts and percentages
# reflecting the classification performance, helping users evaluate the best model
# for image classification. These statistics are based on counts and percentages
# derived from the results dictionary passed to the function.
# The statistics returned include total images, number of dog images, number
# of correct matches, breed accuracy, and other metrics related to correct classifications.

def calculates_results_stats(results_dic):
    """
    Computes various statistics based on the classifier's performance in classifying
    images of pets, returning these statistics in a dictionary. These stats help assess
    how well the classifier performs on dog and non-dog images and how accurate it is
    in identifying breeds.

    Parameters:
    results_dic - Dictionary where each key is an image filename and the corresponding value
                  is a list containing:
                  [0]: Pet image label (string)
                  [1]: Classifier label (string)
                  [2]: 1 if labels match, 0 otherwise
                  [3]: 1 if pet image is a dog, 0 otherwise
                  [4]: 1 if classifier predicted a dog, 0 otherwise

    Returns:
    results_stats_dic - Dictionary containing statistics with:
                        n_images: total number of images
                        n_dogs_img: number of dog images
                        n_notdogs_img: number of non-dog images
                        n_match: number of label matches
                        n_correct_dogs: correctly classified dog images
                        n_correct_notdogs: correctly classified non-dog images
                        n_correct_breed: correctly classified dog breeds
                        pct_match: percentage of correct matches
                        pct_correct_dogs: percentage of correctly classified dog images
                        pct_correct_breed: percentage of correctly classified breeds
                        pct_correct_notdogs: percentage of correctly classified non-dogs
    """
    # Initialize the results statistics dictionary
    results_stats_dic = {}

    # Initialize counters for dogs, matches, correct dog classifications, and correct breed matches
    results_stats_dic['n_dogs_img'] = 0
    results_stats_dic['n_match'] = 0
    results_stats_dic['n_correct_dogs'] = 0
    results_stats_dic['n_correct_notdogs'] = 0
    results_stats_dic['n_correct_breed'] = 0

    # Loop through each result entry in results_dic
    for key in results_dic:
        # Count exact label matches
        if results_dic[key][2] == 1:
            results_stats_dic['n_match'] += 1
        
        # Count correct breed matches: image is a dog and labels match
        if sum(results_dic[key][2:]) == 3:
            results_stats_dic['n_correct_breed'] += 1

        # Count the number of images that are dogs
        if results_dic[key][3] == 1:
            results_stats_dic['n_dogs_img'] += 1
            
            # Count correctly classified dogs
            if results_dic[key][4] == 1:
                results_stats_dic['n_correct_dogs'] += 1
        else:
            # Count correctly classified non-dogs
            if results_dic[key][4] == 0:
                results_stats_dic['n_correct_notdogs'] += 1

    # Calculate the total number of images
    results_stats_dic['n_images'] = len(results_dic)

    # Calculate the number of non-dog images
    results_stats_dic['n_notdogs_img'] = results_stats_dic['n_images'] - results_stats_dic['n_dogs_img']

    # Calculate percentage of correctly matched labels
    results_stats_dic['pct_match'] = (results_stats_dic['n_match'] / results_stats_dic['n_images']) * 100.0

    # Calculate percentage of correctly classified dog images
    results_stats_dic['pct_correct_dogs'] = (results_stats_dic['n_correct_dogs'] / results_stats_dic['n_dogs_img']) * 100.0 if results_stats_dic['n_dogs_img'] > 0 else 0.0

    # Calculate percentage of correctly classified dog breeds
    results_stats_dic['pct_correct_breed'] = (results_stats_dic['n_correct_breed'] / results_stats_dic['n_dogs_img']) * 100.0 if results_stats_dic['n_dogs_img'] > 0 else 0.0

    # Calculate percentage of correctly classified non-dog images
    results_stats_dic['pct_correct_notdogs'] = (results_stats_dic['n_correct_notdogs'] / results_stats_dic['n_notdogs_img']) * 100.0 if results_stats_dic['n_notdogs_img'] > 0 else 0.0

    # Return the statistics dictionary
    return results_stats_dic
