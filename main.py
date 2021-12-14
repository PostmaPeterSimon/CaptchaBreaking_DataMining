from preprocessing import *
from prediction import *

def main():
    score = []
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")

    # for metric in ["euclidean", "minkowski", "manhattan", "seuclidean"]: #"seuclidean"
    #             model = Classifier(training_dataset,training_lables,metric)
    #             #Execute Test data
    #             score.append(predict_Captha(test_dataset,test_lables,model))
    # print(score)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(0,len(training_dataset)):
        processedImage = preprocess(training_dataset[i], image_height=100, blur=(5,5), dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
        determineTrainingData(processedImage)
        saveTrainingData(training_lables[i])
    for lable in os.listdir("t_data"):
        training_dataset = load_images_from_folder("t_data/"+lable)
        trainingClassifier(training_dataset,lable,"euclidean")
if __name__ == "__main__":
    main()