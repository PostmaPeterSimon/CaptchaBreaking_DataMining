from enum import Enum
from dataclasses import dataclass
import numpy as np
import sys
import shutil

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
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

    if os.path.exists("t_data"):
        shutil.rmtree("t_data")

    for i in range(len(training_dataset)):
        processedImage = preprocess(training_dataset[i], image_height=image_height, blur=blur_tuple, \
                dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
        determineTrainingData(processedImage)
        saveTrainingData(training_lables[i])

    training_dataset.clear()
    training_lables.clear()
    for lable in os.listdir("t_data"):
        for filename in os.listdir("t_data/"+lable):
            img = cv2.imread(os.path.join("t_data/"+lable,filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                training_dataset.append(img)
                training_lables.append(lable)
                assert len(training_dataset)==len(training_lables)

    return training_dataset, training_lables


def score_k_nearest_neighbour_classifier(training_dataset, training_lables, metric, n_neighbors, \
        image_height, blur_size, dilation_size, erosion_size ):
    accuracy = None
    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    trained_classifier = trainingClassifier(training_dataset, training_lables, metric, n_neighbors)

    for testimage in test_dataset:
        processedTestImage = preprocess(testimage, image_height=image_height, blur=blur_tuple, \
            dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
        determineTrainingData(processedTestImage)

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
                training_dataset, training_lables = set_training_data(image_height, blur_size, dilation_size, erosion_size)

                if classifier == Classifiers.K_NEAREST_NEIGHBOUR:
                    for metric in k_nearest_neighbour_metrics:
                        for n_neighbours in range_n_neighbours:
                            accuracy = score_k_nearest_neighbour_classifier(training_dataset, training_lables, \
                                 metric, n_neighbours, image_height, blur_size, dilation_size, erosion_size)
                            scores.append(Classifier_score(classifier=Classifiers.K_NEAREST_NEIGHBOUR, blur = blur_size,\
                                dilation = dilation_size, erosion = erosion_size, metric = metric, \
                                n_neighbours = n_neighbours, accuracy = accuracy))

                elif classifier == Classifiers.MLP_CLASSIFIER:
                    accuracy = score_mlp_classifier(train_data, test_data)
                    scores.append(Classifier_score(Classifiers.MLP_CLASSIFIER, blur = blur_size,\
                                dilation = dilation_size, erosion = erosion_size, accuracy=accuracy))
    
    return scores