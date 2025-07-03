from email import message
from typing import List
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from h11 import Data
import joblib
from pydantic import BaseModel
import numpy as np

from core import model_manager
from core.enum import ModelTypeEnum

router = APIRouter(prefix='/ml', tags=['Machine Learning'])

class predictRequest(BaseModel):
    data: List[List[float]]


class multiLinearRequest(BaseModel):
    DISTANCE: int
    ELAPSED_TIME: float
    SCHEDULED_TIME: float
    DEPARTURE_DELAY: float


class multiLinearBatchRequest(BaseModel):
    data: List[multiLinearRequest]


@router.post('/linearRegression')
def predict_linear_regression(request: predictRequest):    
    model_path = model_manager.get_latest_model(ModelTypeEnum.LinearRegression)
    if not model_path:
        return JSONResponse(content={"code": -1, "message": "Model not found."})
    
    try:
        model = joblib.load(model_path)
        X_input = np.array(request.data)
        print(X_input)
        y_pred = model.predict(X_input)
        print(y_pred.tolist())

        return JSONResponse(content={"code": 0, "message:": "Done successfully.", "prediction": y_pred.tolist()})
    except Exception as ex:
        return JSONResponse(content={"code": -1, "message": f"Prediction failed: {str(ex)}"})


@router.post('/multiLinearRegression')
def predict_multi_linear_regression(request: multiLinearBatchRequest):
    model_path = model_manager.get_latest_model(ModelTypeEnum.MultiLinearRegression)
    if not model_path:
        return JSONResponse(content={"code": -1, "message": "Model not found."})
    
    try:
        model = joblib.load(model_path)
        X_input = np.array([
            [row.DISTANCE, row.ELAPSED_TIME, row.SCHEDULED_TIME, row.DEPARTURE_DELAY] 
            for row in request.data
        ])

        print(X_input)
        y_pred = model.predict(X_input)
        print(y_pred.tolist())

        return JSONResponse(content={"code": 0, "message": "Done successfully.", "Data": y_pred.tolist()})

    except Exception as ex:
        return JSONResponse(content={"code": -1, "message": f"Prediction failed: {str(ex)}"})



# @router.get('/multi-linear-regression')
# def predict_multi_linear_regression(model_type: int = Query(...)):
#     return JSONResponse(content={"code": 0, "message": "Predicted using Multi-Linear Regression"})

# @router.get('/lasso')
# def predict_lasso(model_type: int = Query(...)):
#     return JSONResponse(content={"code": 0, "message": "Predicted using Lasso"})

# @router.get('/ridge')
# def predict_ridge(model_type: int = Query(...)):
#     return JSONResponse(content={"code": 0, "message": "Predicted using Ridge Regression"})