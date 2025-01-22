import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Importing logging functionality and custom exceptions from the src module
from src.logger.logging import logging
from src.exception.exception import CreditFraudException
from src.constants import *

# Importing entity classes for data ingestion artifacts and configuration
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

# Class for handling data ingestion processes
class Data_Ingestion:
    def __init__(self):
        """
        Initializes the Data_Ingestion class and sets up the data ingestion configuration.
        Logs the start of the data ingestion process.
        Raises CreditFraudException if there is an error during initialization.
        """
        try:
            # Log the start of the data ingestion process
            logging.info(f"{'> ' * 10} Data Ingestion {' <' * 10}")
            # Initialize the data ingestion configuration
            self.data_ingestion_config = DataIngestionConfig()

        except Exception as e:
            # Raise a custom exception if any error occurs during initialization
            raise CreditFraudException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by reading the dataset, 
        splitting it into training and testing sets, and saving these sets 
        to specified paths. 

        Returns:
            DataIngestionArtifact: An object containing paths to the 
            training, testing, and raw datasets.

        Raises:
            CreditFraudException: If there is an error during the data ingestion process.
        """
        try:
            # Read the input dataset from a CSV file
            df = pd.read_csv(f"{os.path.join(os.getcwd())}/dataset/fraudTrain.csv")
            logging.info("Read the input dataset")
            
            # Split the dataset into training and testing sets
            train_data, test_data = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            logging.info("Performed split operation for train and test dataset")

            # Create the directory for data ingestion if it doesn't exist
            os.makedirs(self.data_ingestion_config.dataset_dir, exist_ok=True)
            logging.info("Created the folder for data ingestion")

            # Save the raw, training, and testing datasets to their respective paths
            df.to_csv(self.data_ingestion_config.raw_path, index=False, header=True)
            train_data.to_csv(self.data_ingestion_config.train_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_path, index=False, header=True)
            logging.info("Saved the train and test datasets")

            # Return the paths of the saved datasets
            return (self.data_ingestion_config.train_path,
                    self.data_ingestion_config.test_path,
                    self.data_ingestion_config.raw_path)

        except Exception as e:
            # Raise a custom exception if any error occurs during data ingestion
            raise CreditFraudException(e, sys)
