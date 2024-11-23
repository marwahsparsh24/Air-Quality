# src/app.py
import os
import flask
import pickle
from google.cloud import storage
import pandas as pd
import pickle5 as pickle

app = flask.Flask(__name__)

# Global model variable
model = None

def load_model():
    global model
    if model is None:
        bucket_name = "airquality-mlops-rg"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        model_path = f'weights/model/model.pkl'
        blob = bucket.blob(model_path)
        model_str = blob.download_as_string()
        model = pickle.loads(model_str)
        # model_save_path = 'model.pkl'
        # with open(model_save_path, 'rb') as f:
        #     model = pickle.load(f)

load_model()

@app.route('/health', methods=['GET'])
def health():
    return {"status": "healthy"}

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if flask.request.content_type != 'application/json':
        return flask.make_response(
            'Expected application/json content-type', 400)

    try:
        instances = flask.request.json["instances"]
        df = pd.DataFrame(instances)
        predictions = model.predict(df).tolist()
        return flask.jsonify({"predictions": predictions})
    except Exception as e:
        return flask.make_response(str(e), 400)

def flask_app(request):
    return app(request)