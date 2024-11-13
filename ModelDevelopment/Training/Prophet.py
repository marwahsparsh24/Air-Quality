import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.prophet
import mlflow.pyfunc
import time
import shap


class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, prophet_model):
        self.prophet_model = prophet_model
    
    def predict(self, context, model_input):
        # Assuming model_input is a DataFrame with 'ds' column (date)
        return self.prophet_model.predict(model_input)['yhat']

class ProphetPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.model = Prophet()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
    def load_data(self):
        # Load training and test data
        train_data = pd.read_pickle(self.train_file)
        # Extract Box-Cox transformed y and original y
        for column in train_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_train = train_data[column]
                break
        self.y_train_original = train_data['pm25']
        self.X_train = train_data.drop(columns=['pm25'])

    def preprocess_data(self):
        # Prepare training data in Prophet format
        self.df_train = pd.DataFrame({
            'ds': self.X_train.index.tz_localize(None),  # Remove timezone
            'y': self.y_train  # Use Box-Cox transformed target
        })

    def train_model(self):
        # Train the Prophet model
        self.model.fit(self.df_train)
        wrapped_model = ProphetWrapper(self.model)


        # local_artifact_path = os.path.join(os.getcwd(), "mlruns", "prophet_pm25_model")
        # os.makedirs(local_artifact_path, exist_ok=True)
        
        # Log the model using the local path
        # mlflow.pyfunc.log_model(
        #     artifact_path=local_artifact_path,
        #     python_model=wrapped_model,
        #     input_example=self.df_train.head(1)
        # )
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        print(mlflow.get_tracking_uri())
        mlflow.pyfunc.log_model(artifact_path="prophet_pm25_model", python_model= wrapped_model,input_example=self.df_train.head(1))

    

    def save_weights(self):
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.log_artifact(self.model_save_path)
        print(f"Model saved at {self.model_save_path}")

# Main function to orchestrate the workflow
def main():

    # os.environ["MLFLOW_TRACKING_URI"] = f"file://{os.path.join(os.getcwd(), 'mlruns')}"
    # os.environ["MLFLOW_ARTIFACT_URI"] = os.path.join(os.getcwd(), "mlruns")
    # os.environ["HOME"] = os.getcwd() 
    # mlruns_dir = os.path.join(os.getcwd(), "mlruns")
    # if not os.path.exists(mlruns_dir):
    #     os.makedirs(mlruns_dir)
    # mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    #mlflow.set_tracking_uri("./mlruns")
    print(os.environ["MLFLOW_TRACKING_URI"])
    # mlruns_path = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("PM2.5 Prophet")


    
    curr_dir = os.getcwd()
    print(curr_dir)
    data_prepocessing_path_pkl = os.path.join(curr_dir,'DataPreprocessing/src/data_store_pkl_files')
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    sys.path.append(data_prepocessing_path_pkl)
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    model_save_path = os.path.join(curr_dir, 'weights/prophet_pm25_model.pth')
    
    if mlflow.active_run():
        mlflow.end_run()
    
    with mlflow.start_run() as run:
        start_time = time.time()
        prophet_model = ProphetPM25Model(train_file, None, None, model_save_path)
        prophet_model.load_data()
        prophet_model.preprocess_data()
        prophet_model.train_model()
        train_duration = time.time() - start_time
        mlflow.log_metric("training_duration", train_duration)
        prophet_model.save_weights()
    mlflow.end_run()
if __name__ == "__main__":
    # path = "/Users/srilakshmikanagala/Desktop/Air/dags"
    # mlflow.set_tracking_uri("file:///opt/airflow/dags/mlruns")
    main()
