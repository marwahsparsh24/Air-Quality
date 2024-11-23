# import flask
# import pandas as pd
# import pickle5 as pickle
# from google.cloud import storage

# import functions_framework

# from markupsafe import escape

# # Initialize Flask app
# app = flask.Flask(__name__)

# # Global model variable
# model = None

# def load_model():
#     """
#     Loads the ML model from Google Cloud Storage bucket.
#     """
#     try:
#         global model
#         if model is None:
#             bucket_name = "airquality-mlops-rg"
#             storage_client = storage.Client()
#             bucket = storage_client.bucket(bucket_name)
#             model_path = 'weights/model/model.pkl'
#             blob = bucket.blob(model_path)
#             model_str = blob.download_as_string()
#             model = pickle.loads(model_str)
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")

# # Load model at startup
# load_model()

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     return {"status": "healthy"}

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Prediction endpoint"""
#     if not flask.request.is_json:
#         return flask.jsonify({"error": "Expected application/json content-type"}), 400

#     try:
#         request_json = flask.request.get_json()
#         if "instances" not in request_json:
#             return flask.jsonify({"error": "Missing 'instances' in request JSON"}), 400

#         instances = request_json["instances"]
#         df = pd.DataFrame(instances)
        
#         if model is None:
#             return flask.jsonify({"error": "Model not loaded properly"}), 500

#         predictions = model.predict(df).tolist()
#         return flask.jsonify({"predictions": predictions}), 200

#     except Exception as e:
#         return flask.jsonify({"error": str(e)}), 400

# def flask_app(request):
#     """
#     Cloud Function entry point
#     """
#     return app.wsgi_app(request.environ, request.get_response())



# @functions_framework.http
# def flask_app(request):
#     """HTTP Cloud Function.
#     Args:
#         request (flask.Request): The request object.
#         <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
#     Returns:
#         The response text, or any set of values that can be turned into a
#         Response object using `make_response`
#         <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
#     """
#     request_json = request.get_json(silent=True)
#     request_args = request.args

#     if request_json and "name" in request_json:
#         name = request_json["name"]
#     elif request_args and "name" in request_args:
#         name = request_args["name"]
#     else:
#         name = "World"
#     return f"Hello {escape(name)}!"

import pandas as pd
import pickle5 as pickle
from google.cloud import storage
from flask import jsonify, Request
import functions_framework


# Global model variable
model = None


def load_model():
    """
    Loads the ML model from Google Cloud Storage bucket.
    """
    try:
        global model
        if model is None:
            bucket_name = "airquality-mlops-rg"
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            model_path = 'weights/model/model.pkl'
            blob = bucket.blob(model_path)
            model_str = blob.download_as_string()
            model = pickle.loads(model_str)
    except Exception as e:
        print(f"Error loading model: {str(e)}")


# Load model at startup
load_model()


@functions_framework.http
def flask_app(request: Request):
    """
    HTTP Cloud Function for handling requests.
    Args:
        request (flask.Request): The request object.
    Returns:
        Response: JSON response based on the path and method.
    """
    # Route to health endpoint
    if request.method == "GET" and request.path == "/health":
        return jsonify({"status": "healthy"}), 200

    # Route to predict endpoint
    elif request.method == "POST" and request.path == "/predict":
        if not request.is_json:
            return jsonify({"error": "Expected application/json content-type"}), 400

        try:
            request_json = request.get_json()
            if "instances" not in request_json:
                return jsonify({"error": "Missing 'instances' in request JSON"}), 400

            instances = request_json["instances"]
            df = pd.DataFrame(instances)

            if model is None:
                return jsonify({"error": "Model not loaded properly"}), 500

            predictions = model.predict(df).tolist()
            return jsonify({"predictions": predictions}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # Default for unsupported paths
    return jsonify({"error": "Unsupported path or method"}), 404

