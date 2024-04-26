import os

import numpy as np
from sktime.base import load as sk_load
from sktime.classification.deep_learning import SimpleRNNClassifier

from gesture_classification.config import RANDOM_SEED
from gesture_classification.modelling.abstract_model import AbstractClassifier


class SkRNNClassifier(AbstractClassifier):
    NAME = "RNN"

    def __init__(
        self,
        n_epochs=100,
        batch_size=128,
        units=16,
        verbose=True,
        random_state: int = RANDOM_SEED,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = SimpleRNNClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            units=units,
            random_state=random_state,
            verbose=verbose,
            loss="categorical_crossentropy",
            # optimizer=Adam(learning_rate=0.001),
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

    def save(self, path: str, **kwargs):
        if os.path.exists(path):
            os.remove(path)

        self.model.save(path, serialization_format="cloudpickle")

    @classmethod
    def load(cls, path, **kwargs) -> "SkRNNClassifier":
        class_instance = cls()
        class_instance.model = sk_load(path)

        return class_instance
