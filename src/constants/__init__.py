## data ingestion
import os

ARTIFACT_DIR = "artifacts"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
RAW_FILE = "CreditCardData.csv"
TARGET = "is_fraud"



SCHEME_PATH = "config/scheme.yaml"

FINAL_PREPROCESSOR_PATH = os.path.join(os.getcwd(),"final-models","preprocessor.pkl")
FINAL_MODEL_PATH = os.path.join(os.getcwd(),"final-models","model.pkl")
PREDICTION_OUTPUT_PATH = os.path.join(os.getcwd(),"prediction-outputs","prediction.csv")

"""
data ingestion
"""
DATA_INGESTION_DIR = "data ingestion"
FEATURE_STORE = "feature store"
DOWNLOAD_ZIP = "data.zip"
TEST_SIZE = 0.2
RANDOM_STATE = 42

"""
data validation
"""
DATA_VALIDATION_DIR = "data validation"
DATA_VALID_DIR = "valid"
DATA_INVALID_DIR = "invalid"
DATA_DRIFT_REPORT_FILE = "report.yml"

"""
data transformation
"""

DATA_TRANSFORMATION_DIR = "data transformation"
TRANSFORMED_TRAIN_FILE = "transformed_train.npy"
TRANSFORMED_TEST_FILE = "transformed_test.npy"
PREPROCESSOR_FILE = "preporcessor.pkl"


"""
Model trainer
"""

MODEL_TRAINER_DIR = "model trainer"
TRAINED_MODEL = "model.pkl"


"""
AWS S3
"""
CREDIT_CARD_DATA_ZIP_FILE = "CreditCardData.zip"
AWS_S3_BUCKET_NAME = "credit-card-fraud-data1"

