from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt


class TRAIN:
    """
    Class for training the  model.
    """

    def __init__(self, model, tuner):
        """
        Initialize the TRAIN class.

        Args:
            model (tf.keras.Model): Model to be trained.
            tuner (kerastuner.engine.tuner.Tuner): Hyperparameter tuner.
        """
        self.model = model
        self.tuner = tuner

    def train_model(self, X_train, Y_train, X_val, Y_val):
        """
        Train the model.

        Args:
            X_train (numpy.ndarray): Training images.
            Y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation images.
            Y_val (numpy.ndarray): Validation labels.

        Returns:
            tf.keras.Model: The trained model.
        """
        # Search for best hyperparameters
        self.tuner.search(x=X_train, y=Y_train,
                          epochs=10,
                          validation_data=(X_val, Y_val))

        # Get the best model from the tuner
        best_model = self.tuner.get_best_models(num_models=1)[0]

        # Compile the best model
        best_model.compile(loss='categorical_crossentropy',
                           optimizer=best_model.optimizer,
                           metrics=['accuracy'])

        callbacks = [
            ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=1e-6),
            TensorBoard(log_dir='./logs', histogram_freq=1)
        ]

        # Train the best model
        history = best_model.fit(X_train, Y_train,
                                 epochs=10,
                                 batch_size=32,
                                 validation_data=(X_val, Y_val),
                                 callbacks=callbacks)

        # Plot training history
        self.plot_training_history(history)

        return best_model

    def plot_training_history(self, history):
        """
        Plot the training history.

        Args:
            history (tf.keras.callbacks.History): Training history object.
        """
        # Plot training & validation accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
