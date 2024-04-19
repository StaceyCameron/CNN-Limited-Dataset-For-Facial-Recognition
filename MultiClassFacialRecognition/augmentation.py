from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AUGMENTATION:
    """
    Class for data augmentation.
    """

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the AUGMENTATION object.

        Args:
            target_size (tuple): Target size for images.
        """
        self.target_size = target_size
        self.label_encoder = LabelEncoder()

    def augment_data(self, X_data, Y_data):
        """
        Augment the data.

        Args:
            X_data (numpy.ndarray): Input images.
            Y_data (numpy.ndarray): Input labels.

        Returns:
            numpy.ndarray: Augmented images.
            numpy.ndarray: Augmented labels.
        """
        # Create an instance of ImageDataGenerator and pass configurations
        data_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Augment the images
        augmented_data = data_generator.flow(X_data, Y_data, batch_size=len(X_data), shuffle=False)

        # Extract augmented images and labels
        X_augmented, Y_augmented = augmented_data.next()

        return X_augmented, Y_augmented
