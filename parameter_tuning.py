from enum import Enum
from dataclasses import dataclass
import shutil
from numpy import degrees
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC
from preprocessing import *

from preprocessing import *
from prediction import *

class Classifiers(Enum):
      K_NEAREST_NEIGHBOUR = 1
      MLP_CLASSIFIER = 2
      SVM_CLASSIFIER = 3

class Classifier_score:
    def __init__(self, classifier, accuracy, blur, dilation, erosion, metric=None, \
                n_neighbours=None, kernel=None, degrees=None, nu=None):
        self.classifier = classifier
        self.accuracy = accuracy
        self.blur = blur
        self.dilation = dilation
        self.erosion = erosion
        self.error = 1-self.accuracy
        self.metric = None if metric == None else metric
        self.n_neighbours = None if n_neighbours == None else n_neighbours
        self.kernel = None if kernel == None else kernel
        self.degrees = None if degrees == None else degrees
        self.nu = None if nu == None else nu


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

def set_testing_data(image_height, blur_size, dilation_size, erosion_size):
    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

    for i in range(len(training_dataset)):
        processedImage = preprocess(training_dataset[i], image_height=image_height, blur=blur_tuple, \
                dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
        determineTrainingData(processedImage)
        saveTrainingData(training_lables[i])


def score_k_nearest_neighbour_classifier(training_dataset, training_lables, metric, n_neighbors,dilation_size, erosion_size,image_height, blur_size):
    #Fitting Classifier
    prediction=[]
    trained_classifier = trainingClassifier(training_dataset, training_lables, metric, n_neighbors)
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    for i in range(len(test_dataset)):
            processedTestImage = preprocess(test_dataset[i], image_height=image_height, blur=blur_tuple, dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
            determineTrainingData(processedTestImage)
            prediction.append(predict_Captha(trained_classifier))
    return makeConfusionMatrix(prediction,test_lables,metric)

def score_svm_classifier(training_dataset, training_lables,svm_kernel, svm_nu, svm_degree,dilation_size, erosion_size,image_height, blur_size):
    #Fitting SVM
    prediction=[]
    training_dataset_as_np_array = np.array(training_dataset)
    nsamples, nx, ny= training_dataset_as_np_array.shape
    training_dataset_as_2d = training_dataset_as_np_array.reshape((nsamples,nx*ny))
    clf = make_pipeline(StandardScaler(), NuSVC(nu=svm_nu, kernel=svm_kernel,degree=svm_degree))
    clf.fit(training_dataset_as_2d, training_lables)
    
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    #Testing SVM
    blur_tuple = (blur_size, blur_size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    for i in range(len(test_dataset)):
        processedTestImage = preprocess(test_dataset[i], image_height=image_height, blur=blur_tuple, dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
        determineTrainingData(processedTestImage)
        prediction.append(predict_Captha(clf))
    return makeConfusionMatrix(prediction,test_lables,svm_kernel)


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
    # SVM_CLASSIFIER
    svm_kernels = {"linear", "poly", "rbf", "sigmoid", "precomputed"}
    svm_degrees = range(0, 6)
    svm_nus = np.arange(0.1, 0.9, 0.1)

    ## Scores ##
    scores = []

    blur_size = 5
    dilation_size = 7
    erosion_size =5
    training_dataset, training_lables = set_training_data(image_height, blur_size, dilation_size, erosion_size)

    if classifier == Classifiers.K_NEAREST_NEIGHBOUR:
        for metric in k_nearest_neighbour_metrics:
            for n_neighbours in range_n_neighbours:
                accuracy = score_k_nearest_neighbour_classifier(training_dataset, training_lables, \
                        metric, n_neighbours, dilation_size, erosion_size,image_height, blur_size)
                scores.append(Classifier_score(classifier=Classifiers.K_NEAREST_NEIGHBOUR, blur = blur_size,\
                    dilation = dilation_size, erosion = erosion_size, metric = metric, \
                    n_neighbours = n_neighbours, accuracy=accuracy))

    elif classifier == Classifiers.SVM_CLASSIFIER:
        for svm_nu in svm_nus:
            for svm_kernel in svm_kernels:
                    for svm_degree in svm_degrees:
                        accuracy = score_svm_classifier(training_dataset, training_lables, svm_kernel, \
                            svm_nu=svm_nu, svm_degree=svm_degree,dilation_size=dilation_size, erosion_size=erosion_size,image_height=image_height, blur_size=blur_size)
                        scores.append(Classifier_score(Classifiers.MLP_CLASSIFIER, blur = blur_size,\
                                    dilation = dilation_size, erosion = erosion_size, accuracy=accuracy))
    return scores