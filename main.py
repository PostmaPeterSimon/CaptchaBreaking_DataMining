import cv2
import os
import numpy as np

# The following function reads the folder with images.
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def find_characters(a_training_dataset):

    for image in a_training_dataset:
        imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert the image to grayscale format
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return None

def main():
    training_dataset = load_images_from_folder("data/training")

if __name__ == "__main__":
    main()