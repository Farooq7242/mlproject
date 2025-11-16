# 📊 Model Information & Comparison

## 🎯 Current Model in Use

**Model Type:** Logistic Regression  
**Location:** `artifacts/model.pkl`  
**Status:** ✅ Active and ready for predictions

---

## 📈 Model Comparison Results

### 🏆 Best Model Selected
- **Model Name:** Logistic Regression
- **Accuracy:** 100.00% (1.0000)
- **Training Date:** 2025-11-16 10:39:47

### 📊 All Models Performance

| Rank | Model | Accuracy |
|------|-------|----------|
| 🏆 1 | Logistic Regression | 100.00% |
| 2 | Random Forest | 100.00% |
| 3 | SVM | 100.00% |
| 4 | Decision Tree | 100.00% |
| 5 | K-Nearest Neighbors | 100.00% |
| 6 | Gradient Boosting | 100.00% |
| 7 | Naive Bayes | 100.00% |

**Total Models Trained:** 7

---

## 📁 File Locations

### Model Files
- **Trained Model:** `artifacts/model.pkl` - Best model saved here
- **Scaler:** `artifacts/scaler.pkl` - Feature scaler for preprocessing
- **Comparison Results:** `artifacts/model_comparison.json` - Complete comparison data

### Log Files
- **Training Logs:** `logs/` folder - Contains all training logs with timestamps

---

## 🔄 How It Works

1. **Training Process:**
   - When you run `python train_models.py`, it trains 7 different models
   - Each model is evaluated on test data
   - All results are compared
   - Best model is automatically selected and saved

2. **Model Selection:**
   - Model with highest accuracy is selected
   - If multiple models have same accuracy, first one is selected
   - Selected model is saved to `artifacts/model.pkl`

3. **Prediction:**
   - Flask app loads the saved model from `artifacts/model.pkl`
   - Uses the same scaler from `artifacts/scaler.pkl`
   - Makes predictions based on user input

---

## 🛠️ Commands

### Check Current Model Info
```bash
python check_model_info.py
```

### Train All Models & Select Best
```bash
python train_models.py
```

### Run Flask App
```bash
python application.py
```

---

## 📝 Notes

- All models achieved 100% accuracy on Iris dataset (this is normal for this simple dataset)
- Logistic Regression was selected as best model (first in ranking)
- Comparison results are saved in JSON format for easy access
- You can view detailed comparison anytime using `check_model_info.py`

