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
def predict_result():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            text=request.form.get('text')
        )
        pred_df = data.get_data()
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    app.run(debug=True, port=8080)
