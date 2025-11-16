import os
import sys
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

class ModelTrainer:
    def __init__(self, model_save_path: str = None):
        """
        Initialize ModelTrainer
        :param model_save_path: Optional custom path to save the trained model
        """
        self.model_path = model_save_path or os.path.join('artifacts', 'model.pkl')
        self.model = None
        self.best_model_name = None

    def get_models(self):
        """
        Returns a dictionary of models to train and compare
        """
        models = {
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        return models

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models, compare their performance, and save the best one
        :param X_train: Training features
        :param X_test: Test features
        :param y_train: Training target
        :param y_test: Test target
        :return: accuracy score of best model
        """
        logger.info(">>> Starting model training with multiple algorithms...")

        try:
            models = self.get_models()
            model_scores = {}
            best_score = 0
            best_model = None
            best_model_name = None

            logger.info(f"Training {len(models)} different models...")
            logger.info("=" * 60)

            # Train each model and evaluate
            for name, model in models.items():
                try:
                    logger.info(f"Training {name}...")
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    
                    model_scores[name] = acc
                    logger.info(f"✅ {name} Accuracy: {acc:.4f}")
                    
                    # Track best model
                    if acc > best_score:
                        best_score = acc
                        best_model = model
                        best_model_name = name
                        
                except Exception as e:
                    logger.warning(f"❌ Error training {name}: {str(e)}")
                    model_scores[name] = 0.0

            logger.info("=" * 60)
            logger.info("📊 Model Comparison Results:")
            logger.info("-" * 60)
            
            # Sort models by accuracy (descending)
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (name, score) in enumerate(sorted_models, 1):
                marker = "🏆" if name == best_model_name else "  "
                logger.info(f"{marker} {i}. {name}: {score:.4f} ({score*100:.2f}%)")

            logger.info("-" * 60)
            logger.info(f"🏆 Best Model: {best_model_name} with accuracy: {best_score:.4f} ({best_score*100:.2f}%)")

            # Save the best model
            self.model = best_model
            self.best_model_name = best_model_name
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_object(self.model_path, self.model)
            logger.info(f"✅ Best model ({best_model_name}) saved successfully at: {self.model_path}")

            # Save model comparison results to JSON file
            comparison_path = os.path.join('artifacts', 'model_comparison.json')
            comparison_data = {
                'best_model': best_model_name,
                'best_accuracy': float(best_score),
                'all_models': {name: float(score) for name, score in model_scores.items()},
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_models_trained': len(models),
                'model_rankings': [
                    {'rank': i+1, 'model': name, 'accuracy': float(score)} 
                    for i, (name, score) in enumerate(sorted_models)
                ]
            }
            
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f, indent=4)
            
            logger.info(f"✅ Model comparison results saved to: {comparison_path}")

            logger.info(">>> Model training process completed successfully.")
            return best_score

        except Exception as e:
            logger.error("❌ Error occurred during model training.")
            raise CustomException(e, sys)
