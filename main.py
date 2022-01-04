from preprocessing import *
from prediction import *
from parameter_tuning import *
from operator import attrgetter

def main():
    score = []
    number_of_neibors= 4
    degrees = 2
    matrix_flag = True
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Make sure that we don't execute training data generation several times
    if not os.path.exists("t_data"):  

        for i in range(len(training_dataset)):
            processedImage = preprocess(training_dataset[i], image_height=100, blur=(5,5), dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
            determineTrainingData(processedImage)
            saveTrainingData(training_lables[i])
 
    training_dataset.clear()
    training_lables.clear()
    for lable in os.listdir("t_data"):
        for filename in os.listdir("t_data/"+lable):
            img = cv2.imread(os.path.join("t_data/"+lable,filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                training_dataset.append(img)
                training_lables.append(lable)
                assert len(training_dataset)==len(training_lables)
    score.append("KNN with "+str(number_of_neibors)+" neighbor")
    for metric in ["manhattan"]: # minkowski
        prediction = []
        model = trainingClassifier(training_dataset,training_lables,metric,number_of_neibors)
        for i in range(len(test_dataset)):
            processedTestImage = preprocess(test_dataset[i], image_height=100, blur=(5,5), dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
            determineTrainingData(processedTestImage)
            prediction.append(predict_Captha(model))
        score.append(makeConfusionMatrix(prediction,test_lables,metric,matrix_flag))
    score.append("SVM with "+str(degrees)+" degrees")
    for svm_kernels in ["poly"]: # "precomputed"
        prediction = [] 
        model = trainingSVM(training_dataset,training_lables,svm_kernels,degrees)
        for i in range(len(test_dataset)):
            processedTestImage = preprocess(test_dataset[i], image_height=100, blur=(5,5), dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
            determineTrainingData(processedTestImage)
            prediction.append(predict_Captha(model))
        score.append(makeConfusionMatrix(prediction,test_lables,svm_kernels,matrix_flag))
    print(score)

def tuning_run():
    knn_scores = tune_and_score_classifiers(Classifiers.K_NEAREST_NEIGHBOUR)
    svm_scores = tune_and_score_classifiers(Classifiers.SVM_CLASSIFIER)
    minimum_knn_error = min(knn_scores, key=attrgetter('error'))
    minimum_svm_error = min(svm_scores, key=attrgetter('error'))
    print(knn_scores)
    print("The best knn is",minimum_knn_error)
    print(svm_scores)
    print("The best svm is",minimum_svm_error)

if __name__ == "__main__":
    main()
    # tuning_run() #Only theoretically but in fact takes a lot of time to finish.


