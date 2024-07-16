from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from joblib import load

from sklearn.preprocessing import MinMaxScaler
from src.components.preprocessor import TextCleaner, TextVectorizer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predict_severity_result():
    if request.method == 'GET':
        return render_template('severity.html', prediction=None, prob_high=None, prob_medium=None, prob_low=None, error=None)
    else:
        try:
            data = CustomData(
                text=request.form.get('text')
            )
            pred_df = data.get_data()
            print(pred_df)

            predict_pipeline = PredictPipeline()

            results, probabilities = predict_pipeline.predict(pred_df)
            prob_high = probabilities[0][2]
            prob_medium = probabilities[0][1]
            prob_low = probabilities[0][0]
            return render_template('severity.html', prediction=results[0].item(), prob_high=prob_high, prob_medium=prob_medium, prob_low=prob_low, error=None)
        except Exception as e:
            return render_template('severity.html', prediction=None, prob_high=None, prob_medium=None, prob_low=None, error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
