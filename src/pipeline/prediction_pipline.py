import os
import sys
import numpy as np
import pandas as pd
from src.exception.exception import CreditFraudException
from src.logger.logging import logging
from src.utils.common import load_object, load_yaml
from src.constants import *

class PredictionPipeline:
    def __init__(self):
        """
        Initialize the Prediction Pipeline by loading configuration settings.
        """
        try:
            self._config = load_yaml(SCHEME_PATH)
            logging.info("Prediction pipeline initialized with configuration.")
        except Exception as e:
            logging.error("Error loading configuration: %s", e)
            raise CreditFraudException(e, sys)

    def predict(self, data):
        """
        Process input data, apply transformations, and make predictions.
        
        Args:
            data (pd.DataFrame): Raw input data.
        
        Returns:
            pd.DataFrame: Data with predicted labels.
        """
        try:
            logging.info("Starting prediction process.")
            df = data.copy()
            data.reset_index(inplace=True, drop=True)
            
            # Drop unwanted columns
            if 'Unnamed: 0' in data.columns:
                data.drop(columns=['Unnamed: 0'], inplace=True)
                logging.info("Dropped 'Unnamed: 0' column.")
            
            # Convert datetime columns
            data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
            data['trans_date'] = data['trans_date_trans_time'].dt.strftime("%Y-%m-%d")
            data['trans_date'] = pd.to_datetime(data['trans_date'])
            data['dob'] = pd.to_datetime(data['dob'])
            
            # Extract date features
            data['trans_month'] = data['trans_date'].dt.month
            data['trans_year'] = data['trans_date'].dt.year
            
            # Compute distance features
            data['latitude_distance'] = abs(round(data['merch_lat'] - data['lat'], 2))
            data['longitude_distance'] = abs(round(data['merch_long'] - data['long'], 2))
            
            # Encode categorical feature
            data['gender'] = data['gender'].replace({'F': 0, 'M': 1}).astype("int64")
            logging.info("Feature engineering completed.")
            
            # Load pre-trained processor and model
            prep_processor = load_object(FINAL_PREPROCESSOR_PATH)
            model = load_object(FINAL_MODEL_PATH)
            logging.info("Loaded preprocessing object and model.")
            
            # Drop unnecessary columns as per configuration
            data = data.drop(columns=self._config.drop_columns, errors='ignore')
            
            # Transform input data
            transformed_data = prep_processor.transform(data)
            logging.info("Data transformation completed.")
            
            # Make predictions
            y_pred = model.predict(transformed_data)
            logging.info("Prediction completed.")
            
            # Store predictions in DataFrame
            df['prediction_output'] = y_pred
            
            # Ensure directory exists before saving output
            dirname = os.path.dirname(PREDICTION_OUTPUT_PATH)
            os.makedirs(dirname, exist_ok=True)
            logging.info("Saving predictions to: %s", PREDICTION_OUTPUT_PATH)
            
            df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
            return df
        
        except Exception as e:
            logging.error("Training Failed :", e)
            raise CreditFraudException(e, sys)
