from preprocessing import *
from prediction import *
from sklearn.metrics import confusion_matrix

def main():
    score = []
    training_dataset = load_images_from_folder("data/training")
    training_lables = get_captha_lable("data/training")
    test_dataset = load_images_from_folder("data/test")
    test_lables = get_captha_lable("data/test")
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Make sure the we don't execute training data generation several times
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

    for metric in ["euclidean", "minkowski", "manhattan"]: #"seuclidean"
        prediction = []
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        model = trainingClassifier(training_dataset,training_lables,metric)
        for i in range(len(test_dataset)):
            processedTestImage = preprocess(test_dataset[i], image_height=100, blur=(5,5), dilation_kernel=dilation_kernel, erosion_kernel=erosion_kernel)
            determineTrainingData(processedTestImage)
            c,i = getConfusionMatrix(model,test_lables[i])
            tp +=c
            fp +=i
            #     prediction.append(predict_Captha(model))
            # print(prediction)
        print("Confusion Matrix:")
        print(tp,'|',fn )
        print(fp,'|',tn )
        accuracy = (tp+tn)/(tp+fn+fp+tn)
        print("Accuracy is:",accuracy )
        prediction.append([metric,accuracy])
    print(prediction)
if __name__ == "__main__":
    main()