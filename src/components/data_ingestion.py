import os
import pandas as pd
from src.logger import logger
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self):
        # File paths where processed data will be stored
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'data.csv')

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process...")
        try:
            # ✅ Load local dataset (instead of URL)
            data_path = os.path.join("data", "Iris.csv")
            logger.info(f"Reading dataset from: {data_path}")

            data = pd.read_csv(data_path)
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            # Save raw copy
            data.to_csv(self.raw_data_path, index=False)
            logger.info("Raw data saved successfully")

            # ✅ Train-test split
            logger.info("Train-test split started")
            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            # Save processed files
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logger.info("Data ingestion completed successfully")

            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)
