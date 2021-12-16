import numpy as np
from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

# This function performs KNearestNeighborClassifier on preprocessed image.
def trainingClassifier(images,lables,a_metric):
    d3array = np.array(images)
    dlable = np.array(lables)
    nsamples, nx, ny, x= d3array.shape
    d2_train_dataset = d3array.reshape((nsamples*nx,ny))
    print(d2_train_dataset.shape)
    print(dlable.shape)
    knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance', algorithm='auto', metric=a_metric)
    knn.fit(d3array,lables)
    return knn

def predict_Captha(images,lables,knn):
    preProcessedImages = []
    for image in images:
        preProcessedImages.append(preprocess(image))
    d3array = np.array(preProcessedImages)
    lables = np.array(lables)
    nsamples, nx, ny = d3array.shape
    d2_data = d3array.reshape((nsamples,nx*ny))
    return knn.predict(d2_data)[0],lables[0],knn.score(d2_data, lables)