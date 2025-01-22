import sys
from src.components.data_ingestion import Data_Ingestion
from src.exception.exception import CreditFraudException


if __name__ == "__main__":
    try:

        data_ingestion = Data_Ingestion()
        data_ingestion.initiate_data_ingestion()     

    except Exception as e:
        raise CreditFraudException(e, sys)