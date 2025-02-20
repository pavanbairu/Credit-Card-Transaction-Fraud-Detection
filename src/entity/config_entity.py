import os
import sys
from datetime import datetime

# Importing constants and custom exceptions from the src module
from src.constants import *
from src.exception.exception import CreditFraudException
from src.logger.logging import logging

# Class to configure the training pipeline
class TrainingPipelineConfig:
    def __init__(self):
        # Get the current timestamp in the format "month_day_year_hour_minute"
        time = datetime.now().strftime("%m_%d_%Y_%H_%M")
        # Define the artifact directory path using the current timestamp
        self.artifact_dir = os.path.join(ARTIFACT_DIR)

# Class to configure data ingestion settings
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            # Initialize the training pipeline configuration
            self.training_pipeline_config = training_pipeline_config
            
            # Create the data ingestion directory path
            self.data_ingestion_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_INGESTION_DIR)
            
            # Define paths for training, testing, and raw datasets using constants
            self.train_path = os.path.join(self.data_ingestion_dir, TRAIN_FILE)
            self.test_path = os.path.join(self.data_ingestion_dir, TEST_FILE)
            self.raw_path = os.path.join(self.data_ingestion_dir, RAW_FILE)

        except Exception as e:
            # Raise a custom exception if any error occurs during initialization
            raise CreditFraudException(e, sys)
        


class DataValidationConfig:
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR)
            self.valid_dir = os.path.join(self.data_validation_dir, DATA_VALID_DIR)
            self.in_valid_dir = os.path.join(self.data_validation_dir, DATA_INVALID_DIR)
            self.validation_train_path = os.path.join(self.valid_dir, TRAIN_FILE)
            self.valid_test_path = os.path.join(self.valid_dir, TEST_FILE)
            self.invalid_train_path = os.path.join(self.in_valid_dir, TRAIN_FILE)
            self.invalid_test_path = os.path.join(self.in_valid_dir, TEST_FILE)
            self.report = os.path.join(self.data_validation_dir, DATA_DRIFT_REPORT_FILE)
        except Exception as e:
            # Raise a custom exception if any error occurs during initialization
            raise CreditFraudException(e, sys)
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR)
        self.transformed_train_path = os.path.join(self.data_transformation_dir, TRANSFORMED_TRAIN_FILE)
        self.transformed_test_path = os.path.join(self.data_transformation_dir, TRANSFORMED_TEST_FILE)
        self.preprocesssor_path = os.path.join(self.data_transformation_dir, PREPROCESSOR_FILE)


       
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR)
        self.model_path = os.path.join(self.model_trainer_dir, TRAINED_MODEL)

        