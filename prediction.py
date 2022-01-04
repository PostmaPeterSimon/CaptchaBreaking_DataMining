from cv2 import Algorithm
import numpy as np
from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler


# This function performs KNearestNeighborClassifier on preprocessed image.
def trainingClassifier(images, lables, a_metric, n_neighbors):
    if n_neighbors == None:
        n_neighbors = 5

    d3array = np.array(images)
    nsamples, nx, ny= d3array.shape
    d2_train_dataset = d3array.reshape((nsamples,nx*ny))
    knn = None
    if a_metric == 'seuclidean':
        V = {'V' : np.var(d2_train_dataset, axis=0)}
        knn = KNeighborsClassifier(metric='seuclidean', metric_params=V, \
            n_neighbors=n_neighbors, algorithm='auto', p=2, weights='distance')
    else:
        knn = KNeighborsClassifier(metric=a_metric, n_neighbors=n_neighbors, \
            p=2, algorithm='auto', weights='distance')

    knn.fit(d2_train_dataset,lables)
    return knn

def predict_Captha(knn):
    try:
        d3array = np.array(listOfCharaters)
        nsamples, nx, ny= d3array.shape
        if nsamples == 5:
            d2_test_dataset = d3array.reshape((nsamples,nx*ny))
            prediction = knn.predict(d2_test_dataset)
            listOfCharaters.clear()
            return ''.join([str(elem) for elem in prediction])
            # print("Not enough charaters to make the prediction")
        listOfCharaters.clear()
    except:
        # print("Something went wrong")
        return

def predict_Captha(knn):
    try:
        d3array = np.array(listOfCharaters)
        nsamples, nx, ny= d3array.shape
        if nsamples == 5:
            d2_test_dataset = d3array.reshape((nsamples,nx*ny))
            prediction = knn.predict(d2_test_dataset)
            listOfCharaters.clear()
            return ''.join([str(elem) for elem in prediction])
            # print("Not enough charaters to make the prediction")
        add_Dummy = 5-nsamples
        if add_Dummy < 0:
            for i in range((-1)*add_Dummy):
                listOfCharaters.pop()
        else:
            for i in range(add_Dummy):
                listOfCharaters.append(' ')
        d3array = np.array(listOfCharaters,dtype=object)
        nsamples, nx, ny= d3array.shape
        assert nsamples == 5
        d2_test_dataset = d3array.reshape((nsamples,nx*ny))
        prediction = knn.predict(d2_test_dataset)
        listOfCharaters.clear()
        return ''.join([str(elem) for elem in prediction])
    except:
        # print("Something went wrong")
        listOfCharaters.clear()
        return "     "

def calculate_matrix_accuracy(matrix, lables):
    # The overall accuracy is calculated as the total number of correctly classified pixels (diagonal elements) divided by the total number of test pixels.
    width, height = matrix.shape
    sum_correctly_classified = 0
    for i in range(width):
        for j in range(height):
            if (i == j):
                sum_correctly_classified += matrix[i][j]

    return sum_correctly_classified / lables

def plot_confusion_matrix(cm,y,metric):
    df_cm = pd.DataFrame(cm, index = [i+1 for i in np.unique(y)],columns = [i+1 for i in np.unique(y)])
    fig = plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix_'+metric)
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.savefig('confusion_matrix_'+metric+'.png', format='png')
    plt.close(fig)

def makeConfusionMatrix(prediction,lables,metric,plot_flag):
    p_char = []
    l_char = []
    for i,string in enumerate(prediction):
        for y,char in enumerate(string):
            p_char.append(ord(char))
            l_char.append(ord(lables[i][y]))
    l_char.append(ord(" "))
    p_char.append(ord(" "))
    confusion_matrix = metrics.confusion_matrix(l_char, p_char)
    if plot_flag:
        plot_confusion_matrix(confusion_matrix,l_char,metric)
    return calculate_matrix_accuracy(confusion_matrix,len(l_char))

def trainingSVM(training_dataset,training_lables,svm_kernel,degrees):
    training_dataset_as_np_array = np.array(training_dataset)
    nsamples, nx, ny= training_dataset_as_np_array.shape
    training_dataset_as_2d = training_dataset_as_np_array.reshape((nsamples,nx*ny))
    clf = make_pipeline(StandardScaler(), NuSVC(nu=0.1, kernel=svm_kernel,degree=degrees))
    return clf.fit(training_dataset_as_2d, training_lables)

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