from PIL.Image import NONE
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from imutils import contours
import random

listOfCharaters = []

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
    return 255-end_image

# This function splits one image into multiple images with one character
def split_multiple_characters(image,c,ROI_number, n_characters):
    x,y,w,h = cv2.boundingRect(c)
    split_width = w//n_characters
    for split in range(0,n_characters):
        split_index = x + split * split_width
        ROI = image[y:y+h, split_index : split_index + split_width]
        listOfCharaters.append(ROI)

# This function contains settings for charater detection function.
def determineTrainingData(image):
    cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Tune cv2 parameters
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    ROI_number = 1
    for c in cnts:
        area = cv2.contourArea(c)
        # print("Area is",area)
        if area >= 700 and area <= 9000: # We can adjust these values and see if it gives us any difference
            if area > 7000:
                split_multiple_characters(image,c,ROI_number,5)
            elif area >= 5000 :
                split_multiple_characters(image,c,ROI_number,4)
            elif area >= 3700 :
                split_multiple_characters(image,c,ROI_number,3)
            elif area >= 2200:
                split_multiple_characters(image,c,ROI_number,2)
            elif area >= 700:
                split_multiple_characters(image,c,ROI_number,1)

# This function filters obtained characters into folder for futher training. 
def saveTrainingData(lable):
    for i,l in enumerate(lable):
        dirName = "t_data/"+l
        if not os.path.exists(dirName):
            os.makedirs(dirName)   
        if i > len(listOfCharaters) -1:
            return
        else:
            cv2.imwrite(os.path.join(dirName,'character_{}.png'.format(random.randint(0,999))),listOfCharaters[i])
    listOfCharaters.clear()