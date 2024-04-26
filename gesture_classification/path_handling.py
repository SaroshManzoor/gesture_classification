import os

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
STORAGE_PATH = os.path.join(PROJECT_PATH, "storage")


def get_data_path() -> str:
    """
    Returns the path where extracted date files are supposed to be.
    If the data wasn't yet downloaded/extracted, this will throw an error.

    Note: download_and_extract_data.sh is responsible for creating this
    directory.

    :return: raw extracted data directory path
    """
    path = os.path.join(PROJECT_PATH, "storage", "data")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Extracted data not found where expected. "
            "Make sure <download_and_extract_data.sh> has ran through in "
            "project directory."
        )

    return path


def get_model_registry_path() -> str:
    """
    Returns the path for the model registry.
    Model registry stores the trained model objects.

    :return: model_registry directory path
    """
    registry_path = os.path.join(PROJECT_PATH, "storage", "model_registry")
    os.makedirs(registry_path, exist_ok=True)

    return registry_path


def get_reference_data_path() -> str:
    """
    Returns the directory path for the reference data.
    Reference data is the preprocessed data used for training, saved as numpy
    array.

    :return: reference data directory path
    """
    path = os.path.join(STORAGE_PATH, "reference")
    os.makedirs(path, exist_ok=True)

    return path


def get_monitoring_data_path() -> str:
    """
    Returns the directory path for the data stored for monitoring & evaluatoin

    :return:
    """
    path = os.path.join(STORAGE_PATH, "monitoring")
    os.makedirs(path, exist_ok=True)

    return path


def get_sqlite_db_path() -> str:
    """

    :return:
    """
    return os.path.join(get_monitoring_data_path(), "monitoring_db.sqlite")
