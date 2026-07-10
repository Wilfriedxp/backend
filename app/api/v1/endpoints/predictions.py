"""
predictions.py  —  backend/app/api/v1/endpoints/predictions.py
Four endpoints covering the full ML lifecycle for both models:
  POST /train-return-model   →  train & evaluate the RF classifier
  POST /predict-return       →  run inference for one or more users
  POST /train-traffic-model  →  train & evaluate the RF regressor
  POST /predict-traffic      →  predict tomorrow's visitor count
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.models import (
    ReturnPredictionRequest,
    ReturnPredictionResponse,
    SinglePrediction,
    TrainReturnModelResponse,
    TrainTrafficModelResponse,
    TrafficPredictionResponse,
)
from app.services import ml_service

router = APIRouter()
log    = logging.getLogger("endpoint.predictions")


# ─────────────────────────────────────────────────────────────────────────────
# Return-user model
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/train-return-model",
    response_model=TrainReturnModelResponse,
    summary="Train the return-user Random Forest classifier",
    description=(
        "Trains a Random Forest classifier to predict whether each user will "
        "return within the prediction window.  Requires data to be loaded via "
        "POST /upload first.  Returns accuracy, F1, ROC-AUC, and CV results."
    ),
)
async def train_return_model() -> TrainReturnModelResponse:
    try:
        result = ml_service.train_return_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        log.exception("Return model training failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {exc}",
        )

    # Remove non-serialisable keys from test_metrics
    safe_metrics = {
        k: v for k, v in result["test_metrics"].items()
        if not isinstance(v, (list, dict))
    }

    return TrainReturnModelResponse(
        message=f"Return-user classifier trained on {result['users_trained_on']} users.",
        users_trained_on=result["users_trained_on"],
        test_metrics=safe_metrics,
        cv_summary=result["cv_summary"],
        feature_importances=result["feature_importances"],
    )


@router.post(
    "/predict-return",
    response_model=ReturnPredictionResponse,
    summary="Predict return likelihood for one or more users",
    description=(
        "Accepts a list of user feature vectors and returns a binary "
        "prediction (will_return) with probability and confidence band "
        "for each user."
    ),
)
async def predict_return(request: ReturnPredictionRequest) -> ReturnPredictionResponse:
    try:
        user_dicts = [u.model_dump() for u in request.users]
        raw_preds  = ml_service.predict_return(user_dicts)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        log.exception("Return prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    return ReturnPredictionResponse(
        predictions=[SinglePrediction(**p) for p in raw_preds],
        model_version="1.0.0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Traffic model
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/train-traffic-model",
    response_model=TrainTrafficModelResponse,
    summary="Train the traffic-volume Random Forest Regressor",
    description=(
        "Trains an RF Regressor on daily visitor counts using lag and calendar "
        "features.  Uses real uploaded data when available, otherwise falls back "
        "to a synthetic 365-day dataset.  Returns MAE, RMSE, R², and MAPE."
    ),
)
async def train_traffic_model() -> TrainTrafficModelResponse:
    try:
        result = ml_service.train_traffic_model()
    except Exception as exc:
        log.exception("Traffic model training failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {exc}",
        )

    safe_metrics = {
        k: v for k, v in result["test_metrics"].items()
        if isinstance(v, (int, float, str))
    }

    return TrainTrafficModelResponse(
        message="Traffic forecasting model trained successfully.",
        training_days=result["training_days"],
        test_metrics=safe_metrics,
        cv_summary=result["cv_summary"],
        feature_importances=result["feature_importances"],
    )


@router.post(
    "/predict-traffic",
    response_model=TrafficPredictionResponse,
    summary="Predict tomorrow's visitor count",
    description=(
        "Uses the trained RF Regressor and the most recent 14 days of daily "
        "visit counts to predict tomorrow's traffic, including an 80 % "
        "confidence interval derived from individual tree predictions."
    ),
)
async def predict_traffic() -> TrafficPredictionResponse:
    try:
        result = ml_service.predict_traffic()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        log.exception("Traffic prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    return TrafficPredictionResponse(**result, model_version="1.0.0")
