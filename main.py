from preprocessing import *
import random

def main():
    training_dataset = preprocess.load_images_from_folder("data/training")
    training_lables = preprocess.get_captha_lable("data/training")
    preprocess.Classifier(training_dataset,training_lables)
if __name__ == "__main__":
    main()