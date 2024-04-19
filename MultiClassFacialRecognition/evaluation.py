import numpy as np
import matplotlib.pyplot as plt
from preprocessing import PREPROCESSING


class EVALUATION:
    """
    Class for evaluating the model on unseen data.
    """

    def __init__(self, trained_model, label_encoder, threshold=0.5):
        """
        Initialize the EVALUATION class.

        Args:
            trained_model (tf.keras.Model): Trained model.
            label_encoder (LabelEncoder): Label encoder.
            threshold (float, optional): Confidence threshold for accepting predictions.
        """
        self.model = trained_model
        self.label_encoder = label_encoder
        self.threshold = threshold

    def evaluation_preprocess(self, directory):
        """
        Preprocess unseen images for evaluation.

        Args:
            directory (str): Unseen images directory.

        Returns:
            numpy.ndarray: Preprocessed images.
        """
        preprocessor = PREPROCESSING()  # Initialize preprocessing class
        preprocessor.load_image_files(directory)  # Load and preprocess unseen images
        return preprocessor.X

    def evaluation_test(self, evaluation_images):
        """
        Test the trained model on unseen images.

        Args:
            evaluation_images (numpy.ndarray): Preprocessed unseen images.
        """
        predictions = self.model.predict(evaluation_images)
        predicted_labels = np.argmax(predictions, axis=1)
        confidence_levels = np.max(predictions, axis=1)
        decoded_predicted_labels = self.label_encoder.inverse_transform(predicted_labels)

        num_images = min(6, len(evaluation_images))
        fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
        plt.suptitle("Evaluation Images", fontsize=16)

        for i in range(num_images):
            axes[i].imshow(evaluation_images[i])

            confidence = confidence_levels[i]
            if confidence >= self.threshold:
                predicted_label_string = decoded_predicted_labels[i]
            else:
                predicted_label_string = "Unknown"

            axes[i].set_title(f"Predicted: {predicted_label_string}\nConfidence: {confidence:.2f}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
