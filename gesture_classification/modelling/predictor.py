import os
from typing import List

from gesture_classification.application.response_models import Prediction
from gesture_classification.modelling.abstract_model import AbstractClassifier
from gesture_classification.modelling.classifiers.classifiers import (
    get_classifiers,
)


class Predictor:
    def __init__(self, model_registry_path: str):
        self.model_registry_path = model_registry_path
        self.classifiers = []

    def predict(self, samples) -> List[Prediction]:
        # Loading models here to bypass the error on first startup
        # This could be done in a better way

        self.classifiers = self.load_classifiers()

        predictions = [
            Prediction(
                classifier=classifier.NAME,
                predicted_class=classifier.predict(samples)[0],
            )
            for classifier in self.classifiers
        ]

        return predictions

    def load_classifiers(self) -> List[AbstractClassifier]:
        existing_classifiers = get_classifiers()

        trained_classifiers = []
        for classifier in existing_classifiers:
            classifier_path = os.path.join(
                self.model_registry_path, f"{classifier.NAME}"
            )

            try:
                trained_classifiers.append(classifier.load(classifier_path))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"<{classifier.NAME}> not found in the registry."
                )

        return trained_classifiers
