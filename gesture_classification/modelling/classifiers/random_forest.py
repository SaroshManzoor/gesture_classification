import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from gesture_classification.config import RANDOM_SEED
from gesture_classification.data_handling.preprocessing import (
    calculate_time_steps_per_sample,
)
from gesture_classification.modelling.abstract_model import AbstractClassifier


class RandomForest(AbstractClassifier):
    NAME = "SummaryRandomForest"

    def __init__(self, n_estimators=200, verbose=True, **kwargs):
        super().__init__(**kwargs)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            verbose=verbose,
            random_state=RANDOM_SEED,
            criterion="entropy",
            **kwargs,
        )

    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs):
        self.model.fit(self.summary_features(features), labels)

    def predict(self, features: np.ndarray, **kwargs):
        return self.model.predict(self.summary_features(features))

    def evaluate(self, features, labels, **kwargs) -> float:
        return self.model.score(self.summary_features(features), labels)

    def save(self, path: str, **kwargs):
        if os.path.exists(path):
            os.remove(path)

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path, **kwargs) -> "RandomForest":
        class_instance = cls()
        class_instance.model = joblib.load(path)

        return class_instance

    @staticmethod
    def summary_features(samples: np.ndarray):
        dimension_means = samples.mean(axis=-1)
        dimension_deviations = samples.std(axis=-1)

        time_steps = calculate_time_steps_per_sample(samples)
        if dimension_means.ndim > 1:
            time_steps = time_steps.reshape(-1, 1)

        features = np.concatenate(
            [dimension_means, dimension_deviations, time_steps], axis=-1
        )

        # Single sample case
        if features.ndim == 1:
            features = np.expand_dims(features, 0)

        return features
