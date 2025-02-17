import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from src.exception.exception import CreditFraudException
from src.logger.logging import logging
from src.constants import *
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.components.data_validation import DataValidationArtifact
from src.utils.common import load_yaml, save_numpy_array, save_object
from src.constants import SCHEME_PATH
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        Initializes DataTransformation class.
        Args:
            data_validation_artifact (DataValidationArtifact): Validated data paths and status.
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        """
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self._config = load_yaml(SCHEME_PATH)
        logging.info("DataTransformation instance created.")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values based on data type.
        - Numeric columns → Median
        - Categorical columns → Mode
        - Date columns → Minimum date
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        try:
            logging.info("Handling missing values.")
            null_columns = df.columns[df.isnull().any()].tolist()
            if not null_columns:
                return df
            for col in null_columns:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col].fillna(df[col].median(), inplace=True)
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    df[col].fillna(df[col].min(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error in handling missing values: {e}")
            raise CreditFraudException(e, sys)

    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame without duplicate rows.
        """
        try:
            logging.info("Handling duplicate values.")
            df.drop_duplicates(inplace=True, ignore_index=True)
            return df
        except Exception as e:
            logging.error(f"Error in handling duplicates: {e}")
            raise CreditFraudException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering on the dataset.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with new engineered features.
        """
        try:
            logging.info("Performing feature engineering.")
            df['trans_month'] = pd.DatetimeIndex(df['trans_date']).month
            df['trans_year'] = pd.DatetimeIndex(df['trans_date']).year
            df['latitude_distance'] = abs(round(df['merch_lat'] - df['lat'], 2))
            df['longitude_distance'] = abs(round(df['merch_long'] - df['long'], 2))
            df['gender'] = df['gender'].replace({'F': 0, 'M': 1}).astype("int64")
            df = pd.get_dummies(df, columns=['category'], drop_first=True)
            df[df.filter(like='category_').columns] = df[df.filter(like='category_').columns].astype(int)
            return df
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise CreditFraudException(e, sys)

    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
            cols (list): List of column names to drop.
        Returns:
            pd.DataFrame: DataFrame after dropping specified columns.
        """
        try:
            logging.info(f"Dropping columns: {cols}")
            df = df.drop(columns=cols, axis=1)
            return df
        except Exception as e:
            logging.error(f"Error in dropping columns: {e}")
            raise CreditFraudException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process.
        - Handles missing values & duplicates
        - Applies feature engineering
        - Scales features
        - Saves transformed datasets
        Returns:
            DataTransformationArtifact: Paths to transformed data and preprocessor object.
        """
        try:
            logging.info("Initiating data transformation process.")
            train_data_path = self.data_validation_artifact.valid_train_path if self.data_validation_artifact.validation_status else self.data_validation_artifact.invalid_train_path
            test_data_path = self.data_validation_artifact.valid_test_path if self.data_validation_artifact.validation_status else self.data_validation_artifact.invalid_test_path

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            train_data = self.handle_missing_values(train_data)
            test_data = self.handle_missing_values(test_data)

            train_data = self.handle_duplicates(train_data)
            test_data = self.handle_duplicates(test_data)

            train_data = self.feature_engineering(train_data)
            test_data = self.feature_engineering(test_data)

            train_data = self.drop_columns(train_data, self._config.drop_columns)
            test_data = self.drop_columns(test_data, self._config.drop_columns)

            smote = SMOTE(sampling_strategy="minority")
            X_train, y_train = train_data.drop(columns=['is_fraud']), train_data['is_fraud']
            X_test, y_test = test_data.drop(columns=['is_fraud']), test_data['is_fraud']

            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_resampled)
            X_test_scaled = scaler.transform(X_test)

            train_scaled_arr = np.c_[X_train_scaled, y_resampled]
            test_scaled_arr = np.c_[X_test_scaled, y_test]

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            save_numpy_array(train_scaled_arr, self.data_transformation_config.transformed_train_path)
            save_numpy_array(test_scaled_arr, self.data_transformation_config.transformed_test_path)
            save_object(scaler, self.data_transformation_config.preprocesssor_path)

            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                self.data_transformation_config.transformed_train_path,
                self.data_transformation_config.transformed_test_path,
                self.data_transformation_config.preprocesssor_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CreditFraudException(e, sys)
