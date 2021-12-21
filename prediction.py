from cv2 import Algorithm
import numpy as np
from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# This function performs KNearestNeighborClassifier on preprocessed image.
def trainingClassifier(images, lables, a_metric, n_neighbors=None):
    if n_neighbors == None:
        n_neighbors = 5

    d3array = np.array(images)
    dlable = np.array(lables)
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
        listOfCharaters.clear()
        return "     "
    except:
        # print("Something went wrong")
        return "     "

def plot_confusion_matrix(cm,y):
    df_cm = pd.DataFrame(cm, index = [i+1 for i in np.unique(y)],columns = [i+1 for i in np.unique(y)])
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.show()

def makeConfusionMatrix(prediction,lables):
    p_char = []
    l_char = []
    for i,string in enumerate(prediction):
        for y,char in enumerate(string):
            p_char.append(ord(char))
            l_char.append(ord(lables[i][y]))
    l_char.append(ord(" "))
    p_char.append(ord(" "))
    confusion_matrix = metrics.confusion_matrix(l_char, p_char)
    plot_confusion_matrix(confusion_matrix,l_char)