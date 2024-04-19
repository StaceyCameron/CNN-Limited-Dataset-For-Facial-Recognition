import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from augmentation import AUGMENTATION


class PREPROCESSING:
    """
    Class for preprocessing images.
    """

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the PREPROCESSING class.

        Args:
            target_size (tuple): Target size of the images.
        """
        self.target_size = target_size
        self.X = []
        self.Y = []
        self.image_labels = []
        self.detector = MTCNN()
        self.label_encoder = LabelEncoder()
        self.data_augmentation = AUGMENTATION()

    def load_image_files(self, directory):
        """
        Load images from directory.

        Args:
            directory (str): Path to the directory containing image files.

        Returns:
            tuple: Tuple containing preprocessed images and corresponding labels.
        """
        self.X, self.Y = self.load_faces(directory)
        for i in range(3):
            augmented_X, augmented_Y = self.data_augmentation.augment_data(self.X, self.Y)
            self.X = np.concatenate((self.X, augmented_X))
            self.Y = np.concatenate((self.Y, augmented_Y))
        return self.X, self.Y

    def load_faces(self, directory):
        """
        Load and preprocess images.

        Args:
            directory (str): Directory path.

        Returns:
            tuple: Preprocessed images and labels.
        """
        faces = []
        labels = []
        for label, sub_directory in enumerate(os.listdir(directory)):
            print("Processing directory:", sub_directory)
            self.image_labels.append(sub_directory)
            path = os.path.join(directory, sub_directory)
            for im_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, im_name)
                    face = self.extract_face(img_path)
                    if face is not None:
                        faces.append(face)
                        labels.append(label)
                except Exception as e:
                    print(f"Error processing image {im_name}: {str(e)}")

        # Convert lists to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)

        # Shuffle images and labels
        indices = np.arange(len(faces))
        np.random.shuffle(indices)
        faces = faces[indices]
        labels = labels[indices]

        # Encode labels
        self.Y = to_categorical(labels, num_classes=len(self.image_labels))

        return faces, self.Y

    def extract_face(self, filename, margin=32):
        """
        Extract face from image.

        Args:
            filename (str): Path to the image file.
            margin (int): Margin to add around the detected face.

        Returns:
            numpy.ndarray: Preprocessed face image.
        """
        try:
            image = cv.imread(filename)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert colour space to RGB
            faces = self.detector.detect_faces(image)
            if faces:
                x, y, w, h = faces[0]['box']
                x, y = abs(x), abs(y)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w += 2 * margin
                h += 2 * margin
                face = image[y:y + h, x:x + w]
                face_array = cv.resize(face, self.target_size)  # Standardise image size
                face_array = face_array / 255.0  # Normalize pixel values
                return face_array
            else:
                print(f"No face detected in {filename}")
                return None
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            return None

    def split_data(self, test_size=0.2, val_size=0.2):
        """
        Split the data into training, validation, and testing sets.

        Args:
            test_size (float): Size for test split.
            val_size (float): Size for validation split.

        Returns:
            tuple: Tuple containing training, validation, and testing sets.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size / (1 - test_size),
                                                          random_state=42)

        print("Training Set: ", len(X_train))
        print("Validation Set: ", len(X_val))
        print("Test Set: ", len(X_test))

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def plot_images(self, images, labels, num_images=6):
        """
        Plot sample images with the labels.

        Args:
            images (numpy.ndarray): Array of images.
            labels (numpy.ndarray): Array of labels.
            num_images (int): Number of images to plot.
        """
        fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
        plt.suptitle("Training Images", fontsize=16)
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].set_title(self.image_labels[np.argmax(labels[i])])  # Decoding one-hot encoded label
            axes[i].axis('off')
        plt.show()

    def get_num_classes(self):
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(self.image_labels)
