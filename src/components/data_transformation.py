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
from src.constants import *
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        Initializes DataTransformation class.
        Args:
            data_validation_artifact (DataValidationArtifact): Contains validated data paths and status.
            data_transformation_config (DataTransformationConfig): Configuration settings for data transformation.
        """
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self._config = load_yaml(SCHEME_PATH)

    def get_transformed_pipeline(self, num_features, cat_features):
        """
        Creates a data transformation pipeline for numerical and categorical features.
        Args:
            num_features (list): List of numerical feature names.
            cat_features (list): List of categorical feature names.
        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', TargetEncoder())
        ])

        preprocessor = ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

        
        return preprocessor

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering on the dataset by adding new features.
        Args:
            df (pd.DataFrame): Input dataset.
        Returns:
            pd.DataFrame: Transformed dataset with new features.
        """
        try:
            
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            # Extract date and time separately
            df['trans_date'] = df['trans_date_trans_time'].dt.strftime("%Y-%m-%d")
            df['trans_date'] = pd.to_datetime(df['trans_date'])
            df['dob']=pd.to_datetime(df['dob'])
            df['trans_month'] = pd.DatetimeIndex(df['trans_date']).month
            df['trans_year'] = pd.DatetimeIndex(df['trans_date']).year
            df['latitude_distance'] = abs(round(df['merch_lat'] - df['lat'], 2))
            df['longitude_distance'] = abs(round(df['merch_long'] - df['long'], 2))
            df['gender'] = df['gender'].replace({'F': 0, 'M': 1}).astype("int64")
            
            return df
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise CreditFraudException(e, sys)

    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Drops specified columns from the dataset.
        Args:
            df (pd.DataFrame): Input dataset.
            cols (list): List of column names to be dropped.
        Returns:
            pd.DataFrame: Dataset after dropping specified columns.
        """
        try:

            df = df.drop(columns=cols, axis=1)

            return df
        except Exception as e:
            logging.error(f"Error in dropping columns: {e}")
            raise CreditFraudException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process, including:
        - Handling missing values & duplicates
        - Applying feature engineering
        - Scaling features
        - Saving transformed datasets
        Returns:
            DataTransformationArtifact: Paths to transformed data and preprocessor object.
        """
        try:
            logging.info(f"{'> '*10} Data Transformation Started {' <'*10}")
            train_data_path = self.data_validation_artifact.valid_train_path if self.data_validation_artifact.validation_status else self.data_validation_artifact.invalid_train_path
            test_data_path = self.data_validation_artifact.valid_test_path if self.data_validation_artifact.validation_status else self.data_validation_artifact.invalid_test_path

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Read the train and test validated data")
           
            train_data = self.feature_engineering(train_data)
            test_data = self.feature_engineering(test_data)
            logging.info("Performed feature engineering for train and test data successfully")

            train_data = self.drop_columns(train_data, self._config.drop_columns)
            test_data = self.drop_columns(test_data, self._config.drop_columns)
            logging.info(f"Columns dropped successfully for train and test data. {self._config.drop_columns}")

            X_train, y_train = train_data.drop(columns=[TARGET]), train_data[TARGET]
            X_test, y_test = test_data.drop(columns=[TARGET]), test_data[TARGET]

            num_features = [column for column in X_train.columns if X_train[column].dtype != 'object']
            cat_features = [column for column in X_train.columns if column not in num_features]

            pre_processor = self.get_transformed_pipeline(num_features, cat_features)
            logging.info("Data transformation pipeline created successfully.")
            
            X_train_processed = pre_processor.fit_transform(X_train, y_train)
            X_test_processed = pre_processor.transform(X_test)
            logging.info("Fit transformation pipeline to training data and transformation to testing data")
        
            smote = SMOTE(sampling_strategy="minority")
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            X_test_resampled, y_test_resampled = smote.fit_resample(X_test_processed, y_test)
            logging.info("Applied SMOTE for class balancing.")

            train_scaled_arr = np.c_[X_train_resampled, y_train_resampled]
            test_scaled_arr = np.c_[X_test_resampled, y_test_resampled]

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            save_numpy_array(train_scaled_arr, self.data_transformation_config.transformed_train_path)
            save_numpy_array(test_scaled_arr, self.data_transformation_config.transformed_test_path)
            save_object(pre_processor, self.data_transformation_config.preprocesssor_path)
            save_object(pre_processor, FINAL_PREPROCESSOR_PATH)
            logging.info("saved the transformed train & test array's and preprocessor pickle file")

            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                self.data_transformation_config.transformed_train_path,
                self.data_transformation_config.transformed_test_path,
                self.data_transformation_config.preprocesssor_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CreditFraudException(e, sys)