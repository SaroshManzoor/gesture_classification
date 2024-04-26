import os

import numpy as np
from sktime.base import load as sk_load
from sktime.classification.deep_learning.cnn import CNNClassifier

from gesture_classification.modelling.abstract_model import AbstractClassifier
from gesture_classification.config import RANDOM_SEED, CNN_N_EPOCHS


class SkCNN(AbstractClassifier):
    NAME = "CNN"

    def __init__(
        self,
        n_epochs=CNN_N_EPOCHS,
        verbose=True,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = CNNClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            kernel_size=kernel_size,
            avg_pool_size=avg_pool_size,
            n_conv_layers=n_conv_layers,
            callbacks=None,
            verbose=verbose,
            random_state=RANDOM_SEED,
            **kwargs
        )

    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs):
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray, **kwargs):

        _features = features.copy()

        if features.ndim == 2:
            _features = np.expand_dims(_features, 0)

        return self.model.predict(_features)

    def evaluate(self, features, labels, **kwargs) -> float:
        return self.model.score(features, labels)

    def save(self, path: str, **kwargs) -> None:
        if os.path.exists(path):

            os.remove(path)

        self.model.save(path, serialization_format="cloudpickle")

    @classmethod
    def load(cls, path, **kwargs) -> "SkCNN":
        class_instance = cls()
        class_instance.model = sk_load(path)

        return class_instance
