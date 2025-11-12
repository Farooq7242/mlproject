import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import logger
from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        # Scaler save path
        self.scaler_path = os.path.join('artifacts', 'scaler.pkl')

    def initiate_data_transformation(self, train_path, test_path):
        logger.info(">>> Starting data transformation process...")

        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Split into features and target
            matched_columns = [
                col for col in train_df.columns if col.lower() == "species"
            ]
            if not matched_columns:
                raise CustomException(
                    "Target column 'Species' not found in training dataset", sys
                )

            target_column = matched_columns[0]

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            if target_column not in test_df.columns:
                matched_test_columns = [
                    col for col in test_df.columns if col.lower() == "species"
                ]
                if not matched_test_columns:
                    raise CustomException(
                        "Target column 'Species' not found in test dataset", sys
                    )
                target_column_test = matched_test_columns[0]
            else:
                target_column_test = target_column

            X_test = test_df.drop(columns=[target_column_test])
            y_test = test_df[target_column_test]

            logger.info("Feature and target split completed successfully.")

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            logger.info("Feature scaling applied successfully.")

            # Save the scaler
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            logger.info(f"Scaler saved successfully at: {self.scaler_path}")

            logger.info("✅ Data transformation completed successfully.")
            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            logger.error("❌ Error occurred during data transformation.")
            raise CustomException(e, sys)
