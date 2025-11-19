import os

try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for
except ImportError as exc:
    raise ImportError(
        "Flask is required to run this application. Install it via "
        "'pip install flask' or 'pip install -r requirements.txt'."
    ) from exc

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainingPipeline
from src.logger import logger
from src.exception import CustomException

app = Flask(__name__)
application = app   # WSGI entrypoint for Elastic Beanstalk

# Initialize pipelines
predict_pipeline = PredictPipeline()
training_pipeline = TrainingPipeline()


@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """API endpoint for predictions from form"""
    if request.method == 'GET':
        return redirect(url_for('home'))

    try:
        data = request.form

        sepal_length = float(data.get('sepal_length', 0))
        sepal_width = float(data.get('sepal_width', 0))
        petal_length = float(data.get('petal_length', 0))
        petal_width = float(data.get('petal_width', 0))

        logger.info(
            f"Received input - SepalLength: {sepal_length}, "
            f"SepalWidth: {sepal_width}, PetalLength: {petal_length}, PetalWidth: {petal_width}"
        )

        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = predict_pipeline.predict(features)

        species = prediction.replace('Iris-', '') if isinstance(prediction, str) else prediction

        logger.info(f"Input features: {features} -> Prediction: {species}")

        return render_template(
            'index.html',
            prediction_text=f'Iris Species: {species}',
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            input_values=f'Input: {features}'
        )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """REST API endpoint for predictions"""
    try:
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = predict_pipeline.predict(features)

        species = prediction.replace('Iris-', '') if isinstance(prediction, str) else prediction

        return jsonify({
            'prediction': species,
            'features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Endpoint to trigger model training pipeline"""
    try:
        logger.info("Starting training pipeline via /train endpoint")
        training_pipeline.start_training()
        message = "Training pipeline completed successfully."

        logger.info(message)

        if request.headers.get('Accept') == 'application/json':
            return jsonify({'status': 'success', 'message': message}), 200

        return render_template('index.html', prediction_text=message)

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")

        if request.headers.get('Accept') == 'application/json':
            return jsonify({'status': 'error', 'message': str(e)}), 500

        return render_template('index.html', prediction_text=f'Error: {str(e)}'), 500


# -----------------------------------------
# Elastic Beanstalk requires PORT variable
# -----------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # EB will auto use correct port
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "yes"]
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
