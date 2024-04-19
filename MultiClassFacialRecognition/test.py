import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class TEST:
    """
    Class for testing the model.
    """

    def __init__(self, threshold=0.9):
        """
        Initialize the TEST object.

        Args:
            threshold (float): Confidence threshold for accepting predictions.
        """
        self.threshold = threshold

    def test_model(self, trained_model, X_test, Y_test, label_encoder, num_images=6):
        """
        Test the model.

        Args:
            trained_model (tf.keras.Model): Trained model.
            X_test (numpy.ndarray): Testing images.
            Y_test (numpy.ndarray): Testing labels.
            label_encoder (LabelEncoder): Label encoder.
            num_images (int): Number of images for plot.
        """
        predictions = trained_model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)

        # Decode predicted and ground truth labels
        decoded_predicted_labels = label_encoder.inverse_transform(predicted_labels)
        decoded_ground_truth_labels = label_encoder.inverse_transform(np.argmax(Y_test, axis=1))

        fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
        plt.suptitle("Testing Images", fontsize=16)

        for i in range(num_images):
            axes[i].imshow(X_test[i])

            confidence = np.max(predictions[i])
            if confidence >= self.threshold:
                # Decode predicted label
                predicted_label_string = decoded_predicted_labels[i]
            else:
                # Set label to unknown
                predicted_label_string = "Unknown"

            axes[i].set_title(f"Predicted Label: {predicted_label_string}\nConfidence: {confidence:.2f}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        # Generate classification report
        print("Classification Report:")
        print(classification_report(decoded_ground_truth_labels, decoded_predicted_labels))
