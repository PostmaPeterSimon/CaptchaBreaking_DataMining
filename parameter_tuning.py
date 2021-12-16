from enum import Enum

from preprocessing import *
from prediction import *

class Classifiers(Enum):
      K_NEAREST_NEIGHBOUR = 1
      MLP_CLASSIFIER = 2


def get_training_data(image_height, blur, dilation_kernel, erosion_kernel):
    #preprocess
    return #training_data

def get_testing_data(image_height, blur, dilation_kernel, erosion_kernel):
    # preprocess
    return #testing_data

def score_k_nearest_neighbour_classifier(metric, n_neighbours):
    #train Classifier

    #test Classifier
    return #accuracy of K_nearest_neighbour_classifier

def score_mlp_classifier():
    #train Classifier

    #test Classifier
    return #accuracy of mlp_classifier

def tune_and_score_classifiers(classifier):
    ## Preprocessing: ##
    range_blur = range(1, 16)
    range_dilation = range(1, 16)
    range_erosion = range(1, 16)
    
    ## Classifiers ##
    # k_nearest_neighbour
    k_nearest_neighbour_metrics = {"euclidean", "minkowski", "manhattan", "seuclidean"}
    range_n_neighbours = range(2, 15)
    # MLP

    for blur in range_blur:
        for dilation in range_dilation:
            for erosion in range_erosion:
                #Set Preprocessing parameters
                blur_tuple = (blur, blur)
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion))

                if classifier == Classifiers.K_NEAREST_NEIGHBOUR:
                    for metric in k_nearest_neighbour_metrics:
                        for n_neighbours in range_n_neighbours:
                            score_k_nearest_neighbour_classifier(metric, n_neighbours)
                            

                elif classifier == Classifiers.MLP_CLASSIFIER:
                    score_mlp_classifier()
                 
            

