import os
import sys
import numpy as np
import pandas as pd
import boto3
import zipfile
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Importing logging functionality and custom exceptions from the src module
from src.logger.logging import logging
from src.exception.exception import CreditFraudException
from src.constants import *

# Importing entity classes for data ingestion artifacts and configuration
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.aws.s3_operations import download_data, extract_zip_file

# Create S3 client
s3 = boto3.resource("s3")

# Class for handling data ingestion processes
class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.
        Logs the start of the data ingestion process.
        Raises CreditFraudException if an error occurs.
        """
        try:
            logging.info(f"{'> ' * 10} Data Ingestion Started {' <' * 10}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.error("Error in initializing DataIngestion class.")
            raise CreditFraudException(e, sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by:
        - Downloading the dataset
        - Extracting the dataset
        - Reading the dataset into a DataFrame
        - Splitting it into training and testing sets
        - Saving these sets to specified paths

        Returns:
            DataIngestionArtifact: Contains paths to train, test, and raw datasets.

        Raises:
            CreditFraudException: If an error occurs during the ingestion process.
        """
        try:
            logging.info("Starting data ingestion process.")

            # Step 1: Download and extract data
            zip_file_path = download_data()
            feature_store_path = extract_zip_file(zip_file_path)

            # Step 2: Read extracted dataset
            dataset_path = os.path.join(feature_store_path, "CreditCardData.csv")  
            logging.info(f"Reading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)

            # Step 3: Data cleaning (drop unwanted columns)
            logging.info("Dropping unnecessary columns if present.")
            df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors='ignore', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Step 4: Split dataset into train and test sets
            logging.info("Splitting dataset into train and test sets.")
            train_data, test_data = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

            # Step 5: Create necessary directories
            logging.info("Creating directories for saving datasets.")
            os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)

            # Step 6: Save datasets
            logging.info("Saving raw, train, and test datasets.")
            df.to_csv(self.data_ingestion_config.raw_path, index=False)
            train_data.to_csv(self.data_ingestion_config.train_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_path, index=False)

            logging.info("Data ingestion completed successfully.")

            return DataIngestionArtifact(
                train_path=self.data_ingestion_config.train_path,
                test_path=self.data_ingestion_config.test_path,
                raw_path=self.data_ingestion_config.raw_path
            )

        except Exception as e:
            logging.error("Error during data ingestion process.")
            raise CreditFraudException(e, sys)
