import sys
from src.components.data_ingestion import Data_Ingestion 
from src.components.data_transformation import DataTransformation
from src.exception.exception import CreditFraudException
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig, DataTransformationConfig


if __name__ == "__main__":
    try:

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion = Data_Ingestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()     

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,
                                         data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

    except Exception as e:
        raise CreditFraudException(e, sys)