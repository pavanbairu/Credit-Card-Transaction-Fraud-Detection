from fastapi import FastAPI
from src.pipeline.training_pipeline import TrainingPipeline
import uvicorn


# Initialize FastAPI app
app = FastAPI(title="Swagger API", description="FastAPI Swagger Example", version="1.0")

# Home route
@app.get("/", tags=["Home"])
def read_root():
    return {"message": "Welcome to FastAPI Swagger API"}

# Get item by ID
@app.get("/train")
def train_data():

    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()

    return "Training is Successful"

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
