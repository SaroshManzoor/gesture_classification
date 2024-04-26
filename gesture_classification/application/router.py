from typing import List

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi import Request

from gesture_classification import monitoring
from gesture_classification.application.response_models import (
    ModelAccuracy,
    Prediction,
)
from gesture_classification.data_handling.preprocessing import preprocess
from gesture_classification.data_handling.read_data import (
    read_gesture_data_from_bytes,
)

main_router = APIRouter()


@main_router.post(
    path="/train",
    description="Train",
    response_model=List[ModelAccuracy],
)
async def train_all_models(request: Request):
    try:
        model_accuracies = request.app.trainer.train()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Data not found. Make sure data is downloaded in "
            "appropriate location. Refer to README.md for quick start guide",
        )
    monitoring.log_training_metrics(db=request.app.db)

    return model_accuracies


@main_router.post(
    path="/predict",
    description="Predict",
    response_model=List[Prediction],
)
async def predict(request: Request, data_file: UploadFile):
    received_data = await data_file.read()

    try:
        data_sample = preprocess(read_gesture_data_from_bytes(received_data))
        predictions = request.app.predictor.predict(data_sample)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Models not yet trained. Please run the training endpoint first.",
        )
    except AssertionError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid gesture time series."
        )

    monitoring.log_inference_data(
        db=request.app.db, sample=data_sample, predictions=predictions
    )

    return predictions
