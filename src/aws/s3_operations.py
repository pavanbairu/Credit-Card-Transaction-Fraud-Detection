import boto3
import zipfile
import os
import sys
from src.logger.logging import logging
from src.exception.exception import CreditFraudException

# # Create S3 client
s3 = boto3.resource("s3")


def download_object(key, bucket_name, filename):
    """
    Downloads an object from an S3 bucket.
    """
    try:
        logging.info(f"Downloading {key} from bucket {bucket_name} to {filename}")
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(Key=key, Filename=filename)
        logging.info(f"Successfully downloaded {key} to {filename}")
    except Exception as e:
        logging.error("Error in downloading data from S3.")
        raise CreditFraudException(e, sys)

def download_data() -> str:
    """
    Fetches data from S3 and saves it as a zip file.
    Returns the local path of the downloaded zip file.
    """
    try:
        logging.info("Starting data download from S3.")
        zip_download_dir = os.path.join(os.getcwd(), "artifacts", "data_ingestion")
        os.makedirs(zip_download_dir, exist_ok=True)

        zip_file_path = os.path.join(zip_download_dir, "data.zip")
        download_object(key="CreditCardData.zip", bucket_name="credit-card-fraud-data1", filename=zip_file_path)

        logging.info(f"Data successfully downloaded and stored at {zip_file_path}")
        return zip_file_path
    except Exception as e:
        logging.error("Error in downloading data.")
        raise CreditFraudException(e, sys)

def extract_zip_file(zip_file_path: str) -> str:
    """
    Extracts the zip file and returns the extracted directory path.
    """
    try:
        logging.info(f"Extracting data from {zip_file_path}")
        feature_store_path = os.path.join(os.getcwd(), "artifacts", "data_ingestion", "feature_store")
        os.makedirs(feature_store_path, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(feature_store_path)

        logging.info(f"Extraction successful, data available at {feature_store_path}")
        return feature_store_path
    except Exception as e:
        logging.error("Error in extracting zip file.")
        raise CreditFraudException(e, sys)