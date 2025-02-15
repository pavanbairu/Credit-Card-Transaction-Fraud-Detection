import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from src.exception.exception import CreditFraudException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig

from src.components.data_validation import DataValidationArtifact
from src.utils.common import load_yaml, save_numpy_array, save_object
from src.constants import SCHEME_PATH
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

import numpy as np

class DataTransformation:

    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self._config = load_yaml(SCHEME_PATH)

    def handle_missing_values(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Identifies columns with null values and imputes them based on data type.
        - Numeric columns → Median
        - Categorical columns → Mode
        - Date columns → Minimum date in the column
        Returns:
            DataFrame with missing values handled.
        """
        try:
            # Find columns with missing values
            null_columns = df.columns[df.isnull().any()].tolist()

            if not null_columns:
                return df  # Return as is if no missing values

            # Impute missing values
            for col in null_columns:
                if np.issubdtype(df[col].dtype, np.number):  # Numeric columns
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

                elif np.issubdtype(df[col].dtype, np.datetime64):  # Date columns
                    min_date = df[col].min()
                    df[col].fillna(min_date, inplace=True)

                else:  # Categorical columns
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)

            return df

        except Exception as e:
            raise Exception(f"Error in handling missing values: {e}")
        

    def handle_duplicates(self, df) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.
        Returns:
            DataFrame without duplicate rows.
        """
        try:

            df.drop_duplicates(inplace=True, ignore_index=True)
            return df

        except Exception as e:
            raise Exception(f"Error in handling duplicates: {e}")
        

    def feature_engineering(self, df):
        
        df['trans_month'] = pd.DatetimeIndex(df['trans_date']).month
        df['trans_year'] = pd.DatetimeIndex(df['trans_date']).year
        df['latitude_distance'] = abs(round(df['merch_lat'] - df['lat'], 2))
        df['longitude_distance'] = abs(round(df['merch_long']-df['long'], 2))
        df['gender'] = df['gender'].replace({'F': 0, 'M' :1}).astype("int64")
        df = pd.get_dummies(df, columns=['category'], drop_first=True)
        df[df.filter(like='category_').columns] = df[df.filter(like='category_').columns].astype(int)

        return df
    
    def drop_columns(self, df, cols):
        df = df.drop(columns=cols, axis=1)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:

        if self.data_validation_artifact.validation_status:
            train_data = pd.read_csv(self.data_validation_artifact.valid_train_path)
            test_data = pd.read_csv(self.data_validation_artifact.valid_test_path)
        else:
            train_data = pd.read_csv(self.data_validation_artifact.invalid_train_path)
            test_data = pd.read_csv(self.data_validation_artifact.invalid_test_path)

        train_data = self.handle_missing_values(train_data)
        test_data = self.handle_missing_values(test_data)

        train_data = self.handle_duplicates(train_data)
        test_data = self.handle_duplicates(test_data)

        train_data = self.feature_engineering(train_data)
        test_data = self.feature_engineering(test_data)

        train_data = self.drop_columns(train_data, self._config.drop_columns)
        test_data = self.drop_columns(test_data, self._config.drop_columns)

        smote = SMOTE(sampling_strategy="minority")
        
        X_train, y_train = train_data.drop(columns=['is_fraud'], axis=1), train_data['is_fraud']
        X_test, y_test = test_data.drop(columns=['is_fraud'], axis=1), test_data['is_fraud']

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)

        train_scaled_arr = np.c_[X_train_scaled, y_resampled]
        test_sacled_arr = np.c_[X_test_scaled, y_test]

        os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
        save_numpy_array(train_scaled_arr, self.data_transformation_config.transformed_train_path)
        save_numpy_array(test_sacled_arr, self.data_transformation_config.transformed_test_path)
        save_object(scaler, self.data_transformation_config.preprocesssor_path)

        data_transformation_artifact = DataTransformationArtifact(
            self.data_transformation_config.transformed_train_path,
            self.data_transformation_config.transformed_test_path,
            self.data_transformation_config.preprocesssor_path
        )

        return data_transformation_artifact
