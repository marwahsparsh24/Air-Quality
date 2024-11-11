import mlflow
import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from dags.ModelDevelopment.XGBoost import XGBoostPM25Model
from dags.ModelDevelopment.Prophet import ProphetPM25Model
from dags.ModelDevelopment.RandomForest import RandomForestPM25Model

# Model Loading Functions
def load_randomforest_model(filepath):
    """Loads the Random Forest model from the given filepath."""
    randomforest = RandomForestPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
    model_save_path = filepath
    with open(model_save_path, 'rb') as f:
       randomforest = pd.read_pickle(f)
    return randomforest

def load_xgboost_model(filepath):
    """Loads the XGBoost model from the given filepath."""
    xgbmodel = xgb.XGBRegressor(random_state=42)
    xgbmodel.load_model(filepath)
    return xgbmodel

def load_prophet_model(filepath):
    """Loads the Prophet model from the given filepath."""
    prophetmodel = ProphetPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
    with open(filepath, 'rb') as f:
            prophetmodel= pickle.load(f)
    return prophetmodel

# Function to Identify Best Model
def find_best_model_run(experiment_name):
    """Finds the best model run in an experiment based on the lowest RMSE metric."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None, None, None
    
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        print(f"No runs found for experiment '{experiment_name}'.")
        return None, None, None
    
    # Identify the run with the lowest RMSE
    best_run = runs.loc[runs['metrics.RMSE'].idxmin()]
    best_rmse = best_run['metrics.RMSE']
    best_run_id = best_run['run_id']
    
    return best_run_id, best_rmse, experiment_name

# Function to Get Model URI Based on Experiment Type
def get_model_uri_and_loader(experiment_name, best_run_id):
    """Returns the model URI and corresponding loader function based on experiment name."""
    if experiment_name == "PM2.5 Prophet":
        model_uri = f"runs:/{best_run_id}/prophet_pm25_model.pth"
        model_loader = load_prophet_model
    elif experiment_name == "PM2.5 Random Forest":
        model_uri = f"runs:/{best_run_id}/randomforest_pm25_model.pth"
        model_loader = load_randomforest_model
    elif experiment_name == "PM2.5 XGBoost Prediction":
        model_uri = f"runs:/{best_run_id}/xgboost_pm25_model.pth"
        model_loader = load_xgboost_model
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_name}")
    
    return model_uri, model_loader

# Function to Download and Load the Model
def download_and_load_model(model_uri, model_loader):
    """Downloads the model artifact from MLflow and loads it using the provided loader function."""
    model_path = mlflow.artifacts.download_artifacts(model_uri)
    model = model_loader(model_path)
    return model

# Main Function to Get the Best Model Across Experiments
def get_best_model_and_load_weights(experiment_names):
    """Finds and loads the best model across multiple experiments based on RMSE."""
    best_rmse = float('inf')
    best_model = None
    best_experiment_name = None
    best_run_id = None

    for experiment_name in experiment_names:
        run_id, rmse, exp_name = find_best_model_run(experiment_name)
        if run_id is not None and rmse < best_rmse:
            best_rmse = rmse
            best_experiment_name = exp_name
            best_run_id = run_id

    if best_run_id:
        model_uri, model_loader = get_model_uri_and_loader(best_experiment_name, best_run_id)
        best_model = download_and_load_model(model_uri, model_loader)

        print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
        print(f"Validation RMSE: {best_rmse}")
    else:
        print("No valid models found across experiments.")
    
    return best_model, best_rmse, best_experiment_name, best_run_id

# Entry Point Function
def main():
    experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
    model, best_rmse, best_experiment_name, best_run_id = get_best_model_and_load_weights(experiment_names)

    if model:
        print(f"Best Experiment: {best_experiment_name}")
        print(f"Best Run ID: {best_run_id}")
        print(f"Best RMSE: {best_rmse}")
    else:
        print("No model could be loaded.")

if __name__ == "__main__":
    main()
