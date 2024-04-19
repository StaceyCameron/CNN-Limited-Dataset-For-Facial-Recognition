import tensorflow as tf
from keras.applications.mobilenet import MobileNet


class MODEL:
    """
    Class for building the CNN model.
    """

    def __init__(self, num_classes):
        """
        Initialize the MODEL class.

        Args:
            num_classes (int): Number of classes in the dataset.
        """
        self.model = None
        self.num_classes = num_classes

    def build_model(self, hyper_param=None):
        """
        Build the model.

        Args:
            hyper_param (kerastuner.HyperParameters): Hyperparameters for tuning.

        Returns:
            tf.keras.Model: The built facial recognition model.
        """
        base_model = MobileNet(
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_top=False)

        base_model.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        if hyper_param:
            # Set hyperparameters
            hp_learning_rate = hyper_param.Float('learning_rate', min_value=0.001, max_value=0.1, sampling='log')

            # Compile the model with the hyperparameters
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])

        return self.model
