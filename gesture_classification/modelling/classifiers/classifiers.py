from typing import List

from gesture_classification.modelling.abstract_model import AbstractClassifier
from gesture_classification.modelling.classifiers.cnn import SkCNN
from gesture_classification.modelling.classifiers.lstm_fcn import SkLSTMfCN
from gesture_classification.modelling.classifiers.random_forest import (
    RandomForest,
)

CLASSIFIERS = [
    RandomForest,
    SkCNN,
    # SkRNNClassifier, # Takes too long to train & performs poorly
    SkLSTMfCN,
]


def get_initialized_classifiers() -> List[AbstractClassifier]:
    """
    Initializes all implemented classifiers with their default configurations

    :return: List of all initialized models

    """

    return [classifier() for classifier in CLASSIFIERS]


def get_classifiers() -> List[AbstractClassifier]:
    """
    Returns classifiers without initializations

    :return: List of all models objects

    """

    return CLASSIFIERS
