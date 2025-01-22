import os
import sys
from datetime import datetime

# Importing constants and custom exceptions from the src module
from src import constants
from src.exception.exception import CreditFraudException
from src.logger.logging import logging

# Class to configure the training pipeline
class TrainingPipelineConfig:
    def __init__(self):
        # Get the current timestamp in the format "month_day_year_hour_minute"
        time = datetime.now().strftime("%m_%d_%Y_%H_%M")
        # Define the artifact directory path using the current timestamp
        self.artifact_dir = os.path.join("artifacts", time)

# Class to configure data ingestion settings
class DataIngestionConfig:
    def __init__(self):
        try:
            # Initialize the training pipeline configuration
            self.training_pipeline_config = TrainingPipelineConfig()
            
            # Create the data ingestion directory path
            data_ingestion_dir = os.path.join(self.training_pipeline_config.artifact_dir, "data_ingestion")
            
            # Define the dataset directory path
            self.dataset_dir = os.path.join(data_ingestion_dir, "dataset")
            
            # Define paths for training, testing, and raw datasets using constants
            self.train_path = os.path.join(self.dataset_dir, constants.TRAIN_FILE)
            self.test_path = os.path.join(self.dataset_dir, constants.TEST_FILE)
            self.raw_path = os.path.join(self.dataset_dir, constants.RAW_FILE)

        except Exception as e:
            # Raise a custom exception if any error occurs during initialization
            raise CreditFraudException(e, sys)