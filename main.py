from preprocessing import *
import random

def main():
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    # preprocess(random.choice(training_dataset))
    preprocess(training_dataset[0])


if __name__ == "__main__":
    main()