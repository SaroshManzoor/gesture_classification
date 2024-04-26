import os
from io import StringIO
from typing import Union

import pandas as pd

from gesture_classification.path_handling import get_reference_data_path

column_names = ["x", "y", "z"]
separator = " "


def read_gesture_data_from_bytes(data: bytes) -> pd.DataFrame:
    return _read_data(data_ref=StringIO(data.decode("utf-8")))


def read_gesture_data_from_file(path: str) -> pd.DataFrame:
    gesture = pd.DataFrame()

    if "Template_Acceleration" in str(path):
        gesture = _read_data(data_ref=path)

    return gesture


def _read_data(data_ref: Union[str, StringIO]) -> pd.DataFrame:
    return pd.read_csv(
        data_ref,
        names=column_names,
        header=None,
        sep=separator,
    )


def read_reference_data(path: str = None):
    if path is None:
        path = get_reference_data_path()
    try:
        # ToDo: parametrize the reference file name
        reference_data = pd.read_csv(os.path.join(path, "reference_data.csv"),
                                     dtype={"target": str})

        return reference_data
    except FileNotFoundError:
        raise FileNotFoundError(
            "No reference data found for monitoring."
            "Make sure training has run through."
        )
