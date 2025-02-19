import os, sys


from src.exception.exception import CreditFraudException
from src.logger.logging import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_training import ModelTrainer


from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)

class TrainingPipeline:

    def __init__(self):
        """
        Initializes the TrainingPipeline with configuration settings for the entire pipeline.
        """
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Handles the data ingestion process by initializing and running the data ingestion component.

        Args:
            None
        
        Returns:
            DataIngestionArtifact: Contains details of the ingested data.
        """
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise CreditFraudException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Validates the ingested data by initializing and running the data validation component.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing details of the ingested data.
        
        Returns:
            DataValidationArtifact: Contains details of the validation process.
        """
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise CreditFraudException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Transforms the validated data into a format suitable for model training by running the data transformation component.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact containing details of validated data.
        
        Returns:
            DataTransformationArtifact: Contains details of the transformation process.
        """
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact

        except Exception as e:
            raise CreditFraudException(e, sys)

    def start_model_trainer(self, data_transformer_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Trains a machine learning model using the transformed data by running the model trainer component.

        Args:
            data_transformer_artifact (DataTransformationArtifact): Artifact containing transformed data.
        
        Returns:
            ModelTrainerArtifact: Contains details of the training process and the trained model.
        """
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformer_artifact=data_transformer_artifact,
                                         model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        
        except Exception as e:
            raise CreditFraudException(e, sys)
        


    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Executes the entire training pipeline sequentially:
        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training

        Args:
            None
        
        Returns:
            ModelTrainerArtifact: Final artifact from the model training step.
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformer_artifact=data_transformation_artifact)
            logging.info("artifact", model_trainer_artifact)
            return model_trainer_artifact
        
        except Exception as e:
            raise CreditFraudException(e, sys)
