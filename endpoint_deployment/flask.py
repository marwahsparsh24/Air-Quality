# src/app.py
import os
import flask
import pickle
from google.cloud import storage
import pandas as pd

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

@app.before_first_request
def init():
    load_model()

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health():
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('AIP_HTTP_PORT', 8080)))