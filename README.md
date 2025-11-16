# 🌺 Iris Species Classification - ML Project

End-to-end Machine Learning project for Iris species classification built with scikit-learn and Flask. The project automates data ingestion, transformation, model training (with multiple algorithms), and provides a web interface for predictions.

## 🧰 Tech Stack

- **Python 3.8+** (recommended via Conda)
- **Machine Learning:** scikit-learn, pandas, numpy
- **Web Framework:** Flask
- **Models:** Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Gradient Boosting, Naive Bayes
- **Logging and custom exception handling**

## 📋 Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git (for cloning the repository)

## 🚀 Complete Setup Guide

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd mlproject
```

### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n iris-env python=3.8 -y
conda activate iris-env
```

**Option B: Using venv**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Dataset

Ensure `data/Iris.csv` exists with the following columns:
- `Id` (optional)
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species`

## 🎯 Project Workflow

### 1. Train Models (Multiple Algorithms)

Train multiple ML models and automatically select the best one:

```bash
python train_models.py
```

**OR using the pipeline:**

```bash
python -m src.pipeline.train_pipeline
```

**What it does:**
- ✅ Data ingestion → saves `artifacts/data.csv`, `artifacts/train.csv`, `artifacts/test.csv`
- ✅ Data transformation → standard scaling + saves `artifacts/scaler.pkl`
- ✅ Model training → trains 7 different models:
  - Logistic Regression
  - Random Forest
  - SVM (Support Vector Machine)
  - Decision Tree
  - K-Nearest Neighbors
  - Gradient Boosting
  - Naive Bayes
- ✅ Model comparison → compares all models and selects the best one
- ✅ Saves best model → `artifacts/model.pkl`
- ✅ Saves comparison results → `artifacts/model_comparison.json`
- ✅ Logs all activities → `logs/` folder

### 2. Check Model Information

View which model is currently being used and see comparison results:

```bash
python check_model_info.py
```

This will show:
- Current model in use
- Model accuracy
- All models comparison
- Training date

### 3. Run Flask Web Application

Start the web interface for predictions:

```bash
python application.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

**Features:**
- Interactive form for entering flower measurements
- Real-time predictions
- Example values for different species
- Beautiful, responsive UI

**Example Values to Try:**
- **Setosa:** Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2
- **Versicolor:** Sepal Length=6.0, Sepal Width=3.0, Petal Length=4.5, Petal Width=1.5
- **Virginica:** Sepal Length=6.5, Sepal Width=3.0, Petal Length=5.2, Petal Width=2.0

### 4. Test Predictions (Optional)

Test the prediction pipeline with different values:

```bash
python test_predictions.py
```

## 📂 Project Structure

```
mlproject/
├── data/                           # Raw source data
│   └── Iris.csv                   # Iris dataset
│
├── artifacts/                      # Generated outputs
│   ├── data.csv                   # Raw data copy
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── model.pkl                  # Trained best model
│   ├── scaler.pkl                 # Feature scaler
│   └── model_comparison.json      # Model comparison results
│
├── logs/                          # Training logs (auto-generated)
│   └── YYYY_MM_DD_HH_MM_SS.log
│
├── src/                           # Source code
│   ├── components/               # Core components
│   │   ├── data_ingestion.py     # Data loading and splitting
│   │   ├── data_transformation.py # Feature scaling
│   │   └── model_trainer.py      # Model training & comparison
│   │
│   ├── pipeline/                  # Pipeline scripts
│   │   ├── train_pipeline.py     # Training orchestration
│   │   └── predict_pipeline.py   # Prediction pipeline
│   │
│   ├── utils.py                   # Helper functions
│   ├── logger.py                  # Logging configuration
│   └── exception.py              # Custom exceptions
│
├── templates/                     # Flask HTML templates
│   └── index.html                # Prediction form
│
├── application.py                 # Flask web application
├── train_models.py               # Model training script
├── check_model_info.py            # Model info checker
├── test_predictions.py            # Prediction tester
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── MODEL_INFO.md                  # Detailed model information
└── README.md                      # This file
```

## 🔧 Available Commands

| Command | Description |
|--------|-------------|
| `python train_models.py` | Train all models and select best one |
| `python check_model_info.py` | View current model and comparison results |
| `python application.py` | Start Flask web application |
| `python test_predictions.py` | Test predictions with sample data |
| `python -m src.pipeline.train_pipeline` | Run training via pipeline class |

## 📊 Model Comparison

The project automatically trains and compares 7 different ML algorithms:

1. **Logistic Regression** - Linear classifier
2. **Random Forest** - Ensemble method
3. **SVM** - Support Vector Machine
4. **Decision Tree** - Tree-based classifier
5. **K-Nearest Neighbors** - Instance-based learning
6. **Gradient Boosting** - Boosting ensemble
7. **Naive Bayes** - Probabilistic classifier

The best model (highest accuracy) is automatically selected and saved.

## 🌐 API Endpoints

### Web Interface
- `GET /` - Home page with prediction form
- `POST /predict` - Submit form and get prediction

### REST API
- `POST /predict_api` - JSON API for predictions

**Example API Request:**
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": "setosa",
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

## 💻 Python Code Usage

```python
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize pipeline
pipeline = PredictPipeline()

# Make prediction
features = [5.1, 3.5, 1.4, 0.2]  # [SepalLength, SepalWidth, PetalLength, PetalWidth]
prediction = pipeline.predict(features)
print(f"Predicted species: {prediction}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Missing `Species` column**
   - Check CSV headers; they are case-insensitive but must exist
   - Ensure `data/Iris.csv` has the correct format

2. **Model not found error**
   - Run training first: `python train_models.py`
   - Check if `artifacts/model.pkl` exists

3. **Artifacts not updating**
   - Delete the `artifacts/` folder
   - Rerun the training pipeline to regenerate fresh outputs

4. **Port already in use (Flask)**
   - Change port in `application.py`: `app.run(port=5001)`
   - Or stop the process using port 5000

5. **Feature names warning**
   - This is fixed in the current version
   - If you see warnings, ensure you're using the latest code

6. **Same prediction for different values**
   - Check input values in browser console/logs
   - Verify model was trained correctly
   - Try retraining: `python train_models.py`

### Logs Location

- Training logs: `logs/YYYY_MM_DD_HH_MM_SS.log`
- Application logs: Console output when running Flask app

## 📝 File Descriptions

- **`train_models.py`** - Script to train all models and select best one
- **`check_model_info.py`** - Script to view model information and comparison
- **`test_predictions.py`** - Script to test predictions with sample data
- **`application.py`** - Flask web application main file
- **`MODEL_INFO.md`** - Detailed documentation about models

## 🎓 Learning Resources

- Iris Dataset: Classic ML dataset for classification
- Scikit-learn Documentation: https://scikit-learn.org/
- Flask Documentation: https://flask.palletsprojects.com/

## 📄 License

This project is for educational purposes.

## 👥 Contributing

Feel free to fork, modify, and use this project for learning ML pipeline development.

## 🙏 Acknowledgments

- Iris dataset creators
- Scikit-learn team
- Flask framework developers

---

**Happy experimenting! 👩‍🔬👨‍🔬**

For detailed model information, see `MODEL_INFO.md`
