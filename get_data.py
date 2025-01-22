import os
import json
import sys
from dotenv import load_dotenv
import pymongo
import certifi
import numpy as np
import pandas as pd

from src.exception.exception import CreditFraudException
from src.logger.logging import logging

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")

# Ensure we have a valid certificate
ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=ca, connectTimeoutMS=30000, socketTimeoutMS=30000)
        except Exception as e:
            raise CreditFraudException(e, sys)

    def csv_to_json_converter(self, filepath):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        
        except Exception as e:
            raise CreditFraudException(e, sys)
        
    def pushing_data_to_mongo(self, records, database, collections):
        try:
            self.records = records
            self.database = database
            self.collections = collections
            
            # Connect to MongoDB
            db = self.mongo_client[self.database]
            collection = db[self.collections]
            
            # Insert records in batches to avoid timeouts
            batch_size = 10000  # Adjust the batch size as needed
            total_records = len(self.records)
            inserted_count = 0

            for i in range(0, total_records, batch_size):
                batch = self.records[i:i + batch_size]
                try:
                    collection.insert_many(batch, ordered=False)  # `ordered=False` allows continued insertion even if a batch fails
                    inserted_count += len(batch)
                    print(inserted_count)
                except pymongo.errors.BulkWriteError as bwe:
                    logging.error(f"Error inserting batch {i // batch_size + 1}: {bwe.details}")
                    continue  # Skip this batch and proceed with the next
                
            return inserted_count
        
        except Exception as e:
            raise CreditFraudException(e, sys)
    
if __name__ == "__main__":
    FILE_PATH = r"D:\Data Science\github\Projects\ML\Credit-Card-Transaction-Fraud-Detection\dataset\CreditCardData.csv"
    DATABASE = "ml-cluster"
    COLLECTION = "CreditCardData"

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_converter(FILE_PATH)
    no_of_records = networkobj.pushing_data_to_mongo(records, DATABASE, COLLECTION)
    print(f"Number of records inserted: {no_of_records}") 
