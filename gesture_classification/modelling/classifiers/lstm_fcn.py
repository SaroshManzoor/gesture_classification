import os

import numpy as np
from sktime.base import load as sk_load
from sktime.classification.deep_learning import LSTMFCNClassifier

from gesture_classification.config import RANDOM_SEED, LSTM_FCN_N_EPOCHS
from gesture_classification.modelling.abstract_model import AbstractClassifier


class SkLSTMfCN(AbstractClassifier):
    NAME = "LSTM-FCN"

    def __init__(
        self,
        n_epochs=LSTM_FCN_N_EPOCHS,
        batch_size=128,
        verbose=True,
        random_state: int = RANDOM_SEED,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = LSTMFCNClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
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
            try:
                os.remove(path)
            except PermissionError:
                import shutil

                shutil.rmtree(path)

        self.model.save(path, serialization_format="cloudpickle")

    @classmethod
    def load(cls, path, **kwargs) -> "SkLSTMfCN":
        class_instance = cls()
        class_instance.model = sk_load(path)

        return class_instance
