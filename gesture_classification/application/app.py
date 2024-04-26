import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gesture_classification import path_handling
from gesture_classification.application.router import main_router
from gesture_classification.config import RANDOM_SEED
from gesture_classification.modelling.predictor import Predictor
from gesture_classification.modelling.trainer import Trainer
from gesture_classification.monitoring.db import SqliteDatabase
from gesture_classification.monitoring.streamlit.app import MODULE_PATH


def init_application():
    _set_logging()
    _set_seed()

    application = FastAPI(
        debug=True, docs_url="/swagger-ui", lifespan=lifespan
    )
    application.include_router(main_router)

    return application


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.trainer = Trainer(
        data_path=path_handling.get_data_path(),
        model_registry_path=path_handling.get_model_registry_path(),
        test_size=0.2,
    )

    application.predictor = Predictor(
        model_registry_path=path_handling.get_model_registry_path(),
    )

    application.db = SqliteDatabase()

    # Stream-lit setup
    try:
        # This is for running the app from IDE
        current_env = os.environ.copy()

        stream_lit_process = subprocess.Popen(
            [
                "streamlit",
                "run",
                MODULE_PATH,
                "--server.port=8001",
                "--server.headless=true",
                "--server.address=0.0.0.0",
            ],
            env=current_env,
        )

    except ModuleNotFoundError:
        # When running in docker, subprocess won't work nicely.
        # Stream-lit is therefore run as a detached process before running FastApi
        # from the entry-point.

        # Dummy process for consistency
        stream_lit_process = subprocess.Popen(["streamlit", "--version"])

        pass

    yield
    # Terminate stream-lit on tear down
    stream_lit_process.terminate()


def _set_logging():
    import tensorflow as tf
    import warnings
    from pandas.errors import SettingWithCopyWarning

    tf.get_logger().setLevel("ERROR")

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)


def _set_seed():
    # Note: Despite setting the seeds here, sk-time models still might
    # not yield reproducible results
    import tensorflow
    import numpy
    import random
    import torch

    torch.manual_seed(RANDOM_SEED)
    tensorflow.random.set_seed(RANDOM_SEED)
    numpy.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
