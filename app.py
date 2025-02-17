import os
import uvicorn
from fastapi import FastAPI
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger.logging import logging

# Initialize FastAPI app
app = FastAPI(title="Swagger API", description="FastAPI Swagger Example", version="1.0")

# Home route
@app.get("/", tags=["Home"])
def read_root():
    """
    Home endpoint to verify API is running.
    Returns:
        dict: Welcome message.
    """
    logging.info("Home endpoint accessed")
    return {"message": "Welcome to FastAPI Swagger API"}

# Train data route
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
        return "Training is Successful"
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return f"Training failed: {e}"

# Run the app
if __name__ == "__main__":
    logging.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
