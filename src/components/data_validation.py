import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import CreditFraudException
from src.logger.logging import logging

from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initialize DataValidation with ingestion artifacts and config.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Paths for train and test data.
            data_validation_config (DataValidationConfig): Configuration for validation.
        """
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

        self.numeric_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 
                                'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
        
        self.cat_columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 
                            'street', 'city', 'state', 'job', 'dob', 'trans_num']

    def numerical_exists(self, df: pd.DataFrame) -> bool:
        """
        Checks if all required numerical columns exist in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            bool: True if all numeric columns exist, else raises exception.
        """
        try:
            logging.info("Checking for numerical columns existence.")
            expected_columns = df.select_dtypes(exclude='object')
            
            if len(self.numeric_columns) == expected_columns.shape[1]:
                for col in self.numeric_columns:
                    if col not in expected_columns:
                        raise CreditFraudException(f"Missing column: {col}")
                return True
            else:
                raise CreditFraudException("Mismatch in expected numeric columns count.")
        except Exception as e:
            raise CreditFraudException(e, sys)

    def categorical_exists(self, df: pd.DataFrame) -> bool:
        """
        Checks if all required categorical columns exist in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            bool: True if all categorical columns exist, else raises exception.
        """
        try:
            logging.info("Checking for categorical columns existence.")
            expected_columns = df.select_dtypes(include='object')
            
            if len(self.cat_columns) == expected_columns.shape[1]:
                for col in self.cat_columns:
                    if col not in expected_columns:
                        raise CreditFraudException(f"Missing column: {col}")
                return True
            else:
                raise CreditFraudException("Mismatch in expected categorical columns count.")
        except Exception as e:
            raise CreditFraudException(e, sys)

    def drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> tuple:
        """
        Checks for data drift using the KS test.
        
        Args:
            base_df (pd.DataFrame): Reference dataset.
            current_df (pd.DataFrame): New dataset for comparison.
            threshold (float): P-value threshold for drift detection.
        
        Returns:
            tuple: Drift report (dict) and status (bool).
        """
        try:
            logging.info("Checking for data drift.")
            report = {}
            status = True

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                is_found = is_same_dist.pvalue < threshold
                if is_found:
                    status = False
                report[column] = {"p_value": float(is_same_dist.pvalue), "drift_status": bool(is_found)}

            return report, status
        except Exception as e:
            raise CreditFraudException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Runs the data validation process and saves results.
        
        Returns:
            DataValidationArtifact: Artifact containing paths and validation status.
        """
        try:
            logging.info("Initiating data validation process.")
            train_data = pd.read_csv(self.data_ingestion_artifact.train_path)
            test_data = pd.read_csv(self.data_ingestion_artifact.test_path)

            self.numerical_exists(train_data)
            self.numerical_exists(test_data)
            self.categorical_exists(train_data)
            self.categorical_exists(test_data)

            report, status = self.drift(train_data, test_data)
            
            os.makedirs(self.data_validation_config.valid_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.in_valid_dir, exist_ok=True)

            report_path = self.data_validation_config.report
            with open(report_path, "w") as file:
                yaml.dump(report, file, default_flow_style=False)

            if status:
                train_data.to_csv(self.data_validation_config.validation_train_path, index=False)
                test_data.to_csv(self.data_validation_config.valid_test_path, index=False)
                logging.info("Validation successful, data stored in valid directory.")
                return DataValidationArtifact(
                    valid_train_path=self.data_validation_config.validation_train_path,
                    valid_test_path=self.data_validation_config.valid_test_path,
                    invalid_train_path=None,
                    invalid_test_path=None,
                    validation_status=status
                )
            else:
                train_data.to_csv(self.data_validation_config.invalid_train_path, index=False)
                test_data.to_csv(self.data_validation_config.invalid_test_path, index=False)
                logging.warning("Validation failed, data stored in invalid directory.")
                return DataValidationArtifact(
                    valid_train_path=None,
                    valid_test_path=None,
                    invalid_train_path=self.data_validation_config.invalid_train_path,
                    invalid_test_path=self.data_validation_config.invalid_test_path,
                    validation_status=status
                )
        except Exception as e:
            raise CreditFraudException(e, sys)
