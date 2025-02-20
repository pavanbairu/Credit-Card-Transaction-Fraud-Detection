import os
import sys
import numpy as np
import pandas as pd

from src.exception.exception import CreditFraudException
from src.logger.logging import logging
from src.constants import *
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.common import save_object, load_numpy_array
from src.utils.classification_metrics import get_classification_scores

from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    """
    Handles the training of a machine learning model.
    """
    def __init__(self, data_transformer_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class.
        
        Args:
            data_transformer_artifact (DataTransformationArtifact): Data transformation artifact containing paths to transformed data.
            model_trainer_config (ModelTrainerConfig): Configuration entity for model training.
        """
        self.model_trainer_config = model_trainer_config
        self.data_transformer_artifact = data_transformer_artifact

    def train_model(self, X_train, y_train):
        """
        Trains a Logistic Regression model.
        
        Args:
            X_train (np.ndarray): Training feature set.
            y_train (np.ndarray): Training labels.
        
        Returns:
            LogisticRegression: Trained logistic regression model.
        """
        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)

            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise CreditFraudException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Loads transformed data, trains the model, evaluates it, and saves the trained model.
        
        Returns:
            ModelTrainerArtifact: Contains model path and performance metrics.
        """
        try:
            logging.info(f"{'> '*10} Model Trainer Started {' <'*10}")
            train_data = load_numpy_array(self.data_transformer_artifact.transformed_train_path)
            test_data = load_numpy_array(self.data_transformer_artifact.transformed_test_path)
            logging.info("Loading transformed training and testing data.")

            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            model = self.train_model(X_train, y_train)
            logging.info("model training completed.")

            y_train_pred = model.predict(X_train)
            train_metrics = get_classification_scores(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            test_metrics = get_classification_scores(y_test, y_test_pred)
            logging.info("Evaluating model performance.")

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            save_object(model, self.model_trainer_config.model_path)
            save_object(model, FINAL_MODEL_PATH)
            logging.info("Model saved successfully.")

            model_trainer_artifact = ModelTrainerArtifact(
                self.model_trainer_config.model_path,
                train_metrics=train_metrics,
                test_metrics=test_metrics
            )

            logging.info(f"Model training process completed successfully. {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in model training process: {e}")
            raise CreditFraudException(e, sys)
