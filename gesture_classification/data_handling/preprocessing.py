import numpy
import numpy as np
import pandas as pd

from gesture_classification.config import MAX_TIME_STEPS

N_CHANNELS = 3


def verify_sample_integrity(data_samples: list, max_len: int = MAX_TIME_STEPS) -> None:
    for sample in data_samples:
        assert len(sample) == N_CHANNELS

        for channel in sample:
            assert len(channel) == max_len


def validate_time_series(gesture: pd.DataFrame):
    assert len(np.unique(gesture.index.values)) == len(gesture)
    assert isinstance(gesture.index, pd.RangeIndex)
    assert len(gesture.columns) == 3
    assert all([column in ["x", "y", "z"] for column in gesture.columns])


def preprocess(
    gesture: pd.DataFrame, max_time_steps: int = MAX_TIME_STEPS
) -> numpy.ndarray:
    # ToDo: Find a more descriptive name for the method
    """
    Steps:
    - Pads time-series with zeros to MAX_TIME_STEPS.
    - Reshapes to [n_channels, n_time_steps]
    - converts to numpy array

    :param max_time_steps:
    :param gesture:
    :return: ND Array with spatial dimensions along column axis
             and time steps along row axis
    """

    validate_time_series(gesture)

    gesture = gesture.reindex(range(max_time_steps), fill_value=0.0)

    return gesture.transpose().values


def calculate_time_steps_per_sample(samples: np.ndarray):
    # Single sample case
    if samples.ndim == 2:
        samples = np.expand_dims(samples, 0)

    time_steps = np.array([])
    for sample in samples:

        last_non_zero_index_per_axis = np.array([])
        for axis_values in sample:
            last_non_zero_index_per_axis = np.append(
                last_non_zero_index_per_axis, np.nonzero(axis_values)[0][-1]
            )

        time_steps = np.append(time_steps, np.max(last_non_zero_index_per_axis))
    return time_steps
