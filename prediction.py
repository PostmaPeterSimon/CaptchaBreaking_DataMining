import numpy as np
from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

# This function performs KNearestNeighborClassifier on preprocessed image.
def trainingClassifier(images,lables,a_metric):
    d3array = np.array(images)
    dlable = np.array(lables)
    nsamples, nx, ny= d3array.shape
    d2_train_dataset = d3array.reshape((nsamples,nx*ny))
    knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance', algorithm='auto', metric=a_metric)
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

