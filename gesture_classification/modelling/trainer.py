import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from gesture_classification.application.response_models import ModelAccuracy
from gesture_classification.config import RANDOM_SEED, HELD_OUT_USERS
from gesture_classification.data_handling.training import get_training_data
from gesture_classification.modelling.abstract_model import AbstractClassifier
from gesture_classification.modelling.classifiers import (
    get_initialized_classifiers,
)
from gesture_classification.path_handling import (
    get_reference_data_path,
    get_monitoring_data_path,
)


class Trainer:
    def __init__(
        self, data_path: str, model_registry_path: str, test_size: float = 0.3
    ):
        self.data_path = data_path
        self.model_registry_path = model_registry_path
        self.accuracies = {}
        self.test_size = test_size

    def train(self, save_reference_data=True) -> List[ModelAccuracy]:

        file_paths = self.construct_training_data_file_paths()

        samples, labels = get_training_data(file_paths)

        x_train, x_test, y_train, y_test = train_test_split(
            samples,
            labels,
            test_size=self.test_size,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

        if save_reference_data:
            self.save_reference_data(x_train, y_train)

        classifiers = get_initialized_classifiers()

        for classifier in classifiers:
            classifier.train(x_train, y_train)
            self.evaluate(classifier, x_test, y_test)
            self.save_to_registry(classifier)

        # ToDo: Refactor .evaluate method
        pd.Series(self.accuracies).rename(
            "Accuracy"
        ).to_frame().reset_index().to_csv(
            os.path.join(get_monitoring_data_path(), "accuracies.csv"),
        )

        return [
            ModelAccuracy(classifier=classifier_name, accuracy=accuracy)
            for classifier_name, accuracy in self.accuracies.items()
        ]

    def construct_training_data_file_paths(self) -> List[str]:
        file_paths = list(Path(self.data_path).rglob("*.txt"))

        if not len(file_paths):
            raise FileNotFoundError(f"No text file found in {self.data_path}")

        # ToDo: Better way to handle the held-out/validation data set
        # Exclude data from users defined in <HELD_OUT_USERS>, from training data
        path_strings = []
        for path in file_paths:
            for user in HELD_OUT_USERS:
                _path = str(path)
                if f"/{user} " not in _path:
                    path_strings.append(_path)

        return path_strings

    @staticmethod
    def save_reference_data(x_train: np.ndarray, y_train: np.ndarray) -> None:
        reference_data = pd.DataFrame(
            x_train.mean(axis=-1), columns=["Avg-X", "Avg-Y", "Avg-Z"]
        )

        reference_data = pd.concat(
            [
                reference_data,
                pd.DataFrame(
                    x_train.std(axis=-1), columns=["Std-X", "Std-Y", "Std-Z"]
                ),
            ],
            axis=1,
        )

        reference_data.loc[:, "target"] = y_train
        reference_data.to_csv(
            os.path.join(get_reference_data_path(), "reference_data.csv"),
            index=False,
        )

    def evaluate(self, classifier: AbstractClassifier, x_test, y_test) -> None:
        self.accuracies[classifier.NAME] = np.round(
            classifier.evaluate(x_test, y_test) * 100, 4
        )
        # ToDo: Improve consistency of metric evaluation

        class_labels = sorted(np.unique(y_test).tolist())

        predictions = classifier.predict(x_test)
        f1_scores = metrics.f1_score(
            y_true=y_test, y_pred=predictions, average=None
        )
        conf_matrix = metrics.confusion_matrix(
            y_true=y_test, y_pred=predictions
        )

        c_dir = os.path.join(get_monitoring_data_path(), classifier.NAME)
        os.makedirs(c_dir, exist_ok=True)

        # Saving F1 score & Confusion matrix to storage
        # ToDo: Setting labels like this is not cool,
        # ToDo: Improve label-position to label mapping
        pd.DataFrame(
            f1_scores, index=class_labels, columns=[classifier.NAME]
        ).to_csv(os.path.join(c_dir, "f1_score.csv"))

        pd.DataFrame(
            conf_matrix, index=class_labels, columns=class_labels
        ).to_csv(os.path.join(c_dir, "confusion_matrix.csv"))

    def save_to_registry(self, classifier: AbstractClassifier):
        classifier.save(
            os.path.join(self.model_registry_path, classifier.NAME)
        )
