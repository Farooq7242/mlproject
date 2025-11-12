import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object  # Make sure save_object is working correctly

class ModelTrainer:
    def __init__(self, model_save_path: str = None):
        """
        Initialize ModelTrainer
        :param model_save_path: Optional custom path to save the trained model
        """
        self.model_path = model_save_path or os.path.join('artifacts', 'model.pkl')
        self.model = None

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        """
        Train a Logistic Regression model and save it
        :param X_train: Training features
        :param X_test: Test features
        :param y_train: Training target
        :param y_test: Test target
        :return: accuracy score
        """
        logger.info(">>> Starting model training...")

        try:
            # Initialize model
            self.model = LogisticRegression(max_iter=200, random_state=42)
            logger.info("LogisticRegression model initialized.")

            # Fit model
            self.model.fit(X_train, y_train)
            logger.info("Model fitting completed.")

            # Predict
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            logger.info(f"✅ Model Accuracy: {acc:.4f}")

            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_object(self.model_path, self.model)
            logger.info(f"Model saved successfully at: {self.model_path}")

            logger.info(">>> Model training process completed successfully.")
            return acc

        except Exception as e:
            logger.error("❌ Error occurred during model training.")
            raise CustomException(e, sys)
