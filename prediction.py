from cv2 import Algorithm
import numpy as np
from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

# This function performs KNearestNeighborClassifier on preprocessed image.
def trainingClassifier(images,lables,a_metric, n_neighbors=None):
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

def getConfusionMatrix(knn,lable):
    try:
        correct =0
        incorrect = 0
        d3array = np.array(listOfCharaters)
        nsamples, nx, ny= d3array.shape
        d2_test_dataset = d3array.reshape((nsamples,nx*ny))
        prediction = knn.predict(d2_test_dataset)
        listOfCharaters.clear()
        string = ''.join([str(elem) for elem in prediction])
        for i,l in enumerate(lable):
            try:
                c = string[i]
            except:
                c = None
            if c == l:
                correct+=1
            else:
                incorrect+=1
        return correct,incorrect
    except:
        return 0,5

