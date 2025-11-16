import os
import sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.scaler_path = os.path.join('artifacts', 'scaler.pkl')
        self.model = None
        self.scaler = None

    def load_artifacts(self):
        """Load the trained model and scaler"""
        try:
            logger.info("Loading model and scaler artifacts...")
            self.model = load_object(self.model_path)
            self.scaler = load_object(self.scaler_path)
            logger.info("Artifacts loaded successfully")
        except Exception as e:
            raise CustomException(f"Error loading artifacts: {str(e)}", sys)

    def predict(self, features):
        """
        Make prediction on given features
        :param features: List or array of features [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
        :return: Predicted species
        """
        try:
            if self.model is None or self.scaler is None:
                self.load_artifacts()

            # Convert to DataFrame with proper column names to avoid feature names warning
            # Expected features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
            feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
            
            # Check if scaler expects 5 features (with Id) or 4 features (without Id)
            expected_features = self.scaler.n_features_in_
            
            if expected_features == 5:
                # Add dummy Id at the beginning
                features_with_id = [0] + list(features)
                feature_names = ['Id'] + feature_names
                logger.info("Adding dummy Id column for compatibility with trained scaler")
            elif expected_features == 4:
                # Use features as is
                features_with_id = list(features)
            else:
                raise CustomException(
                    f"Unexpected number of features in scaler: {expected_features}. Expected 4 or 5.", sys
                )
            
            # Convert to DataFrame with proper column names
            features_df = pd.DataFrame([features_with_id], columns=feature_names)
            
            # Scale the features (using DataFrame to avoid feature names warning)
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            logger.info(f"Prediction made: {prediction[0]}")
            return prediction[0]
            
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)

