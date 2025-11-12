# ML Project

End-to-end Iris classifier built with scikit-learn. The project automates data ingestion, transformation, training, and artifact storage behind a simple pipeline.

## 🧰 Tech Stack

- Python 3.8 (recommended via Conda)
- pandas, scikit-learn, numpy
- Logging and custom exception handling

## 🚀 Quick Start

1. **Clone & enter the project directory**
   ```bash
   git clone <repo-url>
   cd mlproject
   ```

2. **Create a Conda environment (Python 3.8)**
   ```bash
   conda create -n iris-env python=3.8 -y
   conda activate iris-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the dataset**
   - Ensure `data/Iris.csv` exists (same schema as the classic Iris dataset).
   - The ingestion pipeline copies it into `artifacts/`.

5. **Run the training pipeline**
   ```bash
   python -m src.pipeline.train_pipeline
   ```
   This performs:
   - Data ingestion → saves `artifacts/data.csv`, `artifacts/train.csv`, `artifacts/test.csv`
   - Data transformation → standard scaling + serialized `artifacts/scaler.pkl`
   - Model training (Logistic Regression) → saves `artifacts/model.pkl`
   - Logs accuracy in the console and `src/logs/`.

6. **(Optional) Explore via notebook**
   - Open `ml_notebook.ipynb` for an interactive walk-through: ingestion preview, transformation, training, and sample predictions.

7. **Use the trained model for prediction**
   ```python
   from src.components.model_trainer import ModelTrainer
   import pandas as pd

   # load artifacts if needed
   trainer = ModelTrainer()
   scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
   model = pickle.load(open("artifacts/model.pkl", "rb"))

   sample = pd.read_csv("artifacts/test.csv").drop(columns=["Species"]).head()
   sample_scaled = scaler.transform(sample)
   preds = model.predict(sample_scaled)
   print(preds)
   ```

## 📂 Project Structure

```
├── data/                # raw source data (Iris.csv)
├── artifacts/           # outputs: data splits, scaler, trained model
├── src/
│   ├── components/      # ingestion, transformation, training modules
│   ├── pipeline/        # orchestration scripts
│   ├── utils.py         # common helpers (save/load objects, etc.)
│   ├── logger.py        # central logging config
│   └── exception.py     # custom exception wrapper
├── ml_notebook.ipynb    # exploration & demo notebook
├── requirements.txt
└── README.md
```

## 🛠️ Troubleshooting

- **Missing `Species` column**: check CSV headers; they are case-insensitive but must exist.
- **Model attribute missing**: ensure you call `initiate_model_training` before trying to access `trainer.model`.
- **Artifacts not updating**: delete the `artifacts/` folder and rerun the pipeline to regenerate fresh outputs.

Happy experimenting! 👩‍🔬👨‍🔬