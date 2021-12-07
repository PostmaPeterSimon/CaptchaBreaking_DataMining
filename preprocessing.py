from PIL.Image import NONE
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# This is a class that contains all the pre-processing functions.
class preprocess:
    # The following function reads the folder with images.
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images

    # The following function obtains the captha lable for the file name.
    def get_captha_lable(folder):
        lables = []
        for filename in os.listdir(folder):
            file = os.path.basename(filename)
            lable = file.split(".")[0]
            lables.append(lable)
        return lables


    # This function performs KNearestNeighborClassifier on preprocessed image.
    def Classifier(images,lables):
        preProcessedImages = []
        for image in images:
            preProcessedImages.append(preprocess.preprocess(image))
        d3array = np.array(preProcessedImages)
        nsamples, nx, ny = d3array.shape
        d2_train_dataset = d3array.reshape((nsamples,nx*ny))
        knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform', algorithm='auto')
        knn.fit(d2_train_dataset,lables)
        print(knn.predict(d2_train_dataset))
        print(lables)
        print('Training accuracy score: %.3f' % knn.score(d2_train_dataset, lables))
        

    # The following function transforms the captha image into black and white image.
    def preprocess(image):
        # convert the image to grayscale format 
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,23)
        # make monorom
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # remove noise with the help of contourArea
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                cv2.drawContours(opening, [c], -1, (0,0,0), -1)
        return opening

