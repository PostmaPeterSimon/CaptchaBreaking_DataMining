from enum import Enum
from dataclasses import dataclass
import numpy as np

from preprocessing import *
from prediction import *

class Classifiers(Enum):
      K_NEAREST_NEIGHBOUR = 1
      MLP_CLASSIFIER = 2

@dataclass
class Classifier_score:
    classifier: Classifiers
    blur: int
    dilation: int
    erosion: int
    accuracy: float
    metric: str = "N/A"
    n_neighbors: str = "N/A"


def set_training_data(image_height, blur_size, dilation_size, erosion_size):
    #TODO Delete previous training data

    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

    #determine_training_data

    #TODO preprocess
    return #training_data

def set_testing_data(image_height, blur_size, dilation_size, erosion_size):
    #TODO Delete previous test data 

    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    
    # TODO preprocess
    return #testing_data

def score_k_nearest_neighbour_classifier(metric, n_neighbours):
    accuracy = None
    #train Classifier

    #test Classifier
    return accuracy

def score_mlp_classifier():
    accuracy = None
    #train Classifier

    #test Classifier
    return accuracy

def tune_and_score_classifiers(classifier):
    ## Preprocessing: ##
    range_blur_size = range(1, 16)
    range_dilation_size = range(1, 16)
    range_erosion_size = range(1, 16)
    image_height = 255
    
    ## Classifiers ##
    # k_nearest_neighbour
    k_nearest_neighbour_metrics = {"euclidean", "minkowski", "manhattan", "seuclidean"}
    range_n_neighbours = range(2, 15)
    # MLP

    ## Scores ##
    scores = []

    for blur_size in range_blur_size:
        for dilation_size in range_dilation_size:
            for erosion_size in range_erosion_size:
                train_data = set_training_data(image_height, blur_size, dilation_size, erosion_size)
                test_data = set_testing_data(image_height, blur_size, dilation_size, erosion_size)

                if classifier == Classifiers.K_NEAREST_NEIGHBOUR:
                    for metric in k_nearest_neighbour_metrics:
                        for n_neighbours in range_n_neighbours:
                            accuracy = score_k_nearest_neighbour_classifier(train_data, test_data, metric, n_neighbours)
                            scores.append(Classifier_score(classifier=Classifiers.K_NEAREST_NEIGHBOUR, blur = blur_size,\
                                dilation = dilation_size, erosion = erosion_size, metric = metric, n_neighbours = n_neighbours, accuracy = accuracy))

                elif classifier == Classifiers.MLP_CLASSIFIER:
                    accuracy = score_mlp_classifier(train_data, test_data)
                    scores.append(Classifier_score(Classifiers.MLP_CLASSIFIER, blur = blur_size,\
                                dilation = dilation_size, erosion = erosion_size, accuracy=accuracy))
    
    return scores