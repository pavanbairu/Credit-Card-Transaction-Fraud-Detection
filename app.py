import os
import sys
import pandas as pd
from src.exception.exception import CreditFraudException
from src.pipeline.prediction_pipline import PredictionPipeline
from src.constants import *
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger.logging import logging
from fastapi.templating import Jinja2Templates

# Initialize Jinja2 templates directory
templates = Jinja2Templates(directory="./templates")

# Initialize FastAPI app
app = FastAPI(title="Swagger API", description="FastAPI Swagger Example", version="1.0")

@app.get("/", tags=["Home"])
def read_root():
    """
    Home endpoint to verify API is running.
    Returns:
        dict: Welcome message.
    """
    logging.info("Home endpoint accessed")
    return {"message": "Welcome to FastAPI Swagger API"}

@app.get("/train", tags=["Training"])
def train_data():
    """
    Endpoint to trigger model training.
    Returns:
        str: Training status message.
    """
    try:
        logging.info("Training started")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logging.info("Training completed successfully")
        return {"message": "Training is successful"}
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return {"error": f"Training failed: {e}"}

@app.post("/predict", tags=["Prediction"])
def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Handles prediction for a given CSV file:
    - Reads the uploaded file.
    - Runs prediction pipeline.
    - Returns predictions as an HTML table.

    Args:
        request (Request): FastAPI request object.
        file (UploadFile): Uploaded CSV file containing input data for predictions.

    Returns:
        TemplateResponse: Renders an HTML page with the predictions in a table format.
    """
    try:
        logging.info("Prediction request received")
        df = pd.read_csv(file.file)  # Read uploaded CSV file
        logging.info(f"Uploaded file has {df.shape[0]} rows and {df.shape[1]} columns")

        prediction_pipeline = PredictionPipeline()
        predicted_data = prediction_pipeline.predict(df)
        logging.info("Prediction completed successfully")

        return {"message": "Prediction is successful"}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise CreditFraudException(e, sys)

if __name__ == "__main__":
    logging.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
