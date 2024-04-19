from keras_tuner.src.backend.io import tf
from preprocessing import PREPROCESSING
from tuner import build_tuner
from model import MODEL
from train import TRAIN
from test import TEST
from evaluation import EVALUATION


def main():
    """
    Main function to execute the facial recognition system.
    """
    try:
        # Preprocess the training dataset
        preprocessing = PREPROCESSING(target_size=(224, 224))  # Create instance of preprocessing class
        (preprocessing.load_image_files("C:\\Users\\stace\\PycharmProjects\\HonoursProject\\MultiClassFacialRecognition"
                                        "\\images\\train_lfw"))  # Load images
        X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocessing.split_data()  # Split data
        preprocessing.plot_images(X_train, Y_train)  # Plot sample of images

        # Build the tuner
        tuner = build_tuner(num_classes=preprocessing.get_num_classes())

        # Search for the best hyperparameters
        tuner.search(X_train, Y_train,
                     validation_data=(X_val, Y_val),
                     epochs=10,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

        # Build the model
        model_instance = MODEL(num_classes=preprocessing.get_num_classes())  # Create instance of model class
        model = model_instance.build_model()

        # Train the model
        train = TRAIN(model, tuner)
        trained_model = train.train_model(X_train, Y_train, X_val, Y_val)

        # Test the model
        test = TEST()
        preprocessing.label_encoder.fit(preprocessing.image_labels)
        test.test_model(trained_model, X_test, Y_test, preprocessing.label_encoder)

        # Evaluate the model
        evaluation = EVALUATION(trained_model, preprocessing.label_encoder)
        preprocessed_images = evaluation.evaluation_preprocess("C:\\Users\\stace\\PycharmProjects\\HonoursProject"
                                                               "\\MultiClassFacialRecognition\\images\\"
                                                               "test_expected")
        evaluation.evaluation_test(preprocessed_images)

    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
