from PIL.Image import NONE
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from imutils import contours

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

def resize_image(image, height):
    scale_percent =  image.shape[0] / height
    width = int(image.shape[1] / scale_percent)
    dsize = (width, height)
    print("origional width, height :{}, {}".format(image.shape[1], image.shape[0]))
    print(dsize)
    image = cv2.resize(image, dsize)
    return image

def crop_image(image):
        x,y,w,h = cv2.boundingRect(image)
        image = image[y:y+h,x:x+w]
        return image

    # remove noise with the help of contourArea
def remove_noise(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            cv2.drawContours(image, [c], -1, (0,0,0), -1)
    return image

# The following function transforms the captha image into black and white image.
def preprocess(image, image_height, blur, dilation_kernel, erosion_kernel):
    #dilation kernel must of cv::MorphShapes 
    image = resize_image(image, image_height)

    # convert the image to grayscale format 
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgrayblur = cv2.GaussianBlur(imgray, blur, 0)
    ret,tresh = cv2.threshold(imgrayblur, 0, 255, cv2.THRESH_OTSU)
    noise_removed = remove_noise(tresh)
    
    dilation = cv2.dilate(noise_removed, dilation_kernel, iterations = 1)
    erosion = cv2.erode(dilation, erosion_kernel, iterations = 1)
    

    cropped_image = crop_image(remove_noise(erosion))
    end_image = resize_image(cropped_image, image_height)
    determineTrainingData(end_image)
    return end_image

def determineTrainingData(image):
    cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Tune cv2 parameters
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000 and area < 4700: #add the right area values
            x,y,w,h = cv2.boundingRect(c)
            ROI = 255 - image[y:y+h, x:x+w]
            cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            ROI_number +=1
