from preprocessing import preprocess

def main():
    training_dataset = preprocess.load_images_from_folder("data/training")
    training_lables = preprocess.get_captha_lable("data/training")
    preprocess.preprocess(training_dataset[0])
if __name__ == "__main__":
    main()