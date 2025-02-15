import os
import sys
import numpy as np
import pandas as pd

from src.exception.exception import CreditFraudException
from src.logger import logging
from src.constants import *
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.common import save_object, load_numpy_array
from src.utils.classification_metrics import get_classification_scores

from sklearn.linear_model import LogisticRegression



class ModelTrainer:
    def __init__(self, data_transformer_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig
                 ):
        
        self.model_trainer_config=model_trainer_config
        self.data_transformer_artifact=data_transformer_artifact

    def train_model(self, X_train, y_train):

        model = LogisticRegression()
        model.fit(X_train, y_train)

        return model


    def initiate_model_trainer(self):

        train_data = load_numpy_array(self.data_transformer_artifact.transformed_train_path)
        test_data = load_numpy_array(self.data_transformer_artifact.transformed_test_path)

        X_train, y_train = train_data[:,:-1], train_data[:,-1]
        X_test, y_test = test_data[:,:-1], test_data[:,-1]

        model = self.train_model(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_metrics = get_classification_scores(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        test_metrics = get_classification_scores(y_test, y_test_pred)

        os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
        save_object(model, self.model_trainer_config.model_path)

        model_trainer_artifact = ModelTrainerArtifact(
            self.model_trainer_config.model_path,
            train_metrics=train_metrics,
            test_metrics=test_metrics
        )

        return model_trainer_artifact
