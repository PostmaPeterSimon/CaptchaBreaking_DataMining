from preprocessing import *
import random

def main():
    score = []
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    for metric in ["euclidean", "minkowski", "manhattan"]: #"seuclidean"
                model = Classifier(training_dataset,training_lables,metric)
                #Execute Test data
                score.append(predit_Captha(test_dataset,test_lables,model))
    print(score)
if __name__ == "__main__":
    main()