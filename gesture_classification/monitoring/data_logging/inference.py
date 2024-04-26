from typing import List

import numpy as np
import pandas as pd

from gesture_classification import config
from gesture_classification.application.response_models import Prediction
from gesture_classification.monitoring.db import SqliteDatabase


def log_inference_data(
    db: SqliteDatabase,
    sample: np.ndarray,
    predictions: List[Prediction],
) -> None:
    _log_current_sample(db, sample)
    _log_predictions(db, predictions)


def _log_predictions(
    db: SqliteDatabase, predictions: List[Prediction]
) -> None:
    prediction_table = pd.Series()

    for prediction_object in predictions:
        prediction_table.loc[
            prediction_object.classifier
        ] = prediction_object.predicted_class

    db.insert_data_frame(
        prediction_table.astype("string").to_frame().transpose(),
        table_name=config.PREDICTION_TABLE,
        if_exists="append",
    )


def _log_current_sample(db, sample):
    current_sample = pd.DataFrame(
        [sample.mean(axis=-1)], columns=["Avg-X", "Avg-Y", "Avg-Z"]
    )
    current_sample = pd.concat(
        [
            current_sample,
            pd.DataFrame(
                [sample.std(axis=-1)], columns=["Std-X", "Std-Y", "Std-Z"]
            ),
        ],
        axis=1,
    )
    db.insert_data_frame(
        current_sample,
        table_name=config.INFERENCE_DATA_TABLE,
        if_exists="append",
    )
