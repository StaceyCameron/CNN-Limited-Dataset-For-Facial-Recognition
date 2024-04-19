from keras_tuner import BayesianOptimization
from model import MODEL


def build_tuner(num_classes):
    """
    Build a hyperparameter tuner for the model.

    Args:
        num_classes (int): Number of classes in the dataset.

    Returns:
        kerastuner.engine.tuner.Tuner: Hyperparameter tuner.
    """
    tuner = BayesianOptimization(
        MODEL(num_classes).build_model,
        objective='val_accuracy',
        max_trials=5,
        directory='bayesian_optimization',
        project_name='facial_recognition'
    )
    return tuner