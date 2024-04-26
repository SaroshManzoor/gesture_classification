from typing import Tuple, List

import numpy as np
import pandas as pd

from gesture_classification.config import MIN_TIME_STEPS
from gesture_classification.data_handling import preprocessing
from gesture_classification.data_handling.read_data import (
    read_gesture_data_from_file,
)


def get_training_data(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data_samples = []
    labels = pd.Series(dtype="string")

    for file_index, file_path in enumerate(file_paths):
        gesture = read_gesture_data_from_file(file_path)

        if len(gesture) == 0:
            continue

        sample = preprocessing.preprocess(gesture)
        time_steps = preprocessing.calculate_time_steps_per_sample(sample)[0]

        # Excluding too short samples from training
        if time_steps < MIN_TIME_STEPS:
            continue

        data_samples.append(
            preprocessing.preprocess(gesture),
        )

        labels.loc[file_index] = extract_label_from_filename(file_path)

    preprocessing.verify_sample_integrity(data_samples)

    return np.array(data_samples), labels.values


def extract_label_from_filename(file_name: str) -> str:
    return file_name.split("Template_Acceleration")[1][0]
