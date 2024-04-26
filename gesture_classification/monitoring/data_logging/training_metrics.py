import os

import pandas as pd

from gesture_classification import config
from gesture_classification.modelling.classifiers import get_classifiers
from gesture_classification.monitoring.db import SqliteDatabase
from gesture_classification.path_handling import (
    get_monitoring_data_path,
    get_reference_data_path,
)

MONITORING_DATA_PATH = get_monitoring_data_path()


# ToDo: Split log_training_metrics into smaller functions


def log_training_metrics(db: SqliteDatabase):
    classifiers = get_classifiers()

    f1_scores = []
    confusion_matrices = []

    # Read training metrics from file and log to DB.
    for classifier in classifiers:
        c_dir = os.path.join(MONITORING_DATA_PATH, classifier.NAME)

        # ToDo: parametrize metric file names
        # ToDo: Implement readers and writers for metrics
        f1_scores.append(
            pd.read_csv(os.path.join(c_dir, "f1_score.csv"), index_col=0)
        )

        confusion_matrix = pd.read_csv(
            os.path.join(c_dir, "confusion_matrix.csv"), index_col=0
        )
        # Append classifier name as main index
        confusion_matrix = pd.concat(
            {classifier.NAME: confusion_matrix}, names=["Classifier"]
        )

        confusion_matrices.append(confusion_matrix)

    # Log F1 scores
    f1_table = pd.concat(f1_scores, axis=1).transpose()
    db.insert_data_frame(
        f1_table, table_name=config.F1_TABLE, if_exists="replace"
    )

    # Log Confusion Matrices
    confusion_tables = pd.concat(confusion_matrices, axis=0)
    db.insert_data_frame(
        confusion_tables,
        table_name=config.CONFUSION_TABLE,
        if_exists="replace",
    )

    # Log Confusion Matrices
    accuracies = pd.read_csv(
        os.path.join(MONITORING_DATA_PATH, "accuracies.csv")
    )
    db.insert_data_frame(
        accuracies, table_name=config.ACCURACY_TABLE, if_exists="replace"
    )

    # Log Reference data
    reference_data = pd.read_csv(
        os.path.join(get_reference_data_path(), "reference_data.csv")
    )
    db.insert_data_frame(
        reference_data,
        table_name=config.REFERENCE_DATA_TABLE,
        if_exists="replace",
    )
