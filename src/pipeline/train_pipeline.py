from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException
import sys

class TrainingPipeline:
    def start_training(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(train_path, test_path)

            model_trainer = ModelTrainer()
            acc = model_trainer.initiate_model_training(X_train, X_test, y_train, y_test)

            logger.info(f"Training pipeline completed successfully!")
            logger.info(f"Best Model: {model_trainer.best_model_name}")
            logger.info(f"Best Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            logger.info("===============================================")
        except Exception as e:
            raise CustomException(e, sys)
