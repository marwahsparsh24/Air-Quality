#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle
import numpy as np
from scipy import stats

feature_data_path = f'processed/test/feature_eng_data.pkl'

# Model paths
model_paths = {
    "random_forest": f'weights/rf_model.pth',
    "xgboost": f'weights/xgboost_pm25_model.pth',
    "prophet": f'weights/prophet_pm25_model.pth'
}

# Load feature-engineered test data
def load_feature_data():
    bucket_name = "airquality-mlops-rg"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(feature_data_path)
    pickle_data = blob.download_as_bytes()
    feature_data = pickle.load(BytesIO(pickle_data))
    #feature_data = pd.read_pickle(feature_data_path)
    print("Loaded feature-engineered test data successfully.")

    if 'season' not in feature_data.columns:
        feature_data['season'] = feature_data['month'].apply(lambda x: (
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall'
        ))
        feature_data['season'] = feature_data['season'].astype('category')

    if 'timestamp' not in feature_data.columns:
        feature_data['timestamp'] = pd.date_range(start='1/1/2023', periods=len(feature_data), freq='H')
        print("Warning: 'timestamp' column is missing. Creating a synthetic timestamp column.")

    print("Feature Data Columns:", feature_data.columns)
    return feature_data

# Model loaders
def load_random_forest_model(filepath):
    bucket_name = "airquality-mlops-rg"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filepath)

    temp_file_name = "/tmp/temp_model.pth"
    # Download the file from the bucket to the temporary path
    blob.download_to_filename(temp_file_name)

    try:
        with open(temp_file_name, 'rb') as f:
            model = pd.read_pickle(f)
        print("Model loaded successfully using pandas.read_pickle.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    # with open(filepath, 'rb') as f:
    #     model = pickle.load(f)
    print("Random Forest model loaded successfully.")
    return model

def load_xgboost_model(filepath):
    model = xgb.Booster()
    bucket_name = "airquality-mlops-rg"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filepath)

    temp_file_name = "/tmp/temp_model.pth"
    # Download the file from the bucket to the temporary path
    blob.download_to_filename(temp_file_name)

    try:
        model.load_model(temp_file_name)
        # with open(temp_file_name, 'rb') as f:
        #     self.model = pd.read_pickle(f)
        print("Model loaded successfully using pandas.read_pickle.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    # model.load_model(filepath)
    print("XGBoost model loaded successfully.")
    return model

def load_prophet_model(filepath):
    bucket_name = "airquality-mlops-rg"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filepath)

    temp_file_name = "/tmp/temp_model.pth"
    # Download the file from the bucket to the temporary path
    blob.download_to_filename(temp_file_name)

    try:
        with open(temp_file_name, 'rb') as f:
            model = pd.read_pickle(f)
        print("Model loaded successfully using pandas.read_pickle.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    # with open(filepath, 'rb') as f:
    #     model = pickle.load(f)
    print("Prophet model loaded successfully.")
    return model

def load_models():
    rf_model = load_random_forest_model(model_paths["random_forest"])
    xgb_model = load_xgboost_model(model_paths["xgboost"])
    prophet_model = load_prophet_model(model_paths["prophet"])
    return rf_model, xgb_model, prophet_model

# Make predictions
def make_predictions(feature_data, rf_model, xgb_model, prophet_model):
    X_test = feature_data.drop(columns=['pm25', 'season', 'timestamp'], errors='ignore')
    y_true = feature_data['pm25']

    print("Starting Predictions...")

    try:
        feature_data['y_pred_rf'] = rf_model.predict(X_test)
        print("Random Forest predictions:", feature_data['y_pred_rf'].head())
    except Exception as e:
        print(f"Error in Random Forest prediction: {e}")
        feature_data['y_pred_rf'] = pd.Series([float('nan')] * len(X_test))

    try:
        xgb_test_data = xgb.DMatrix(X_test)
        feature_data['y_pred_xgb'] = xgb_model.predict(xgb_test_data)
        print("XGBoost predictions:", feature_data['y_pred_xgb'][:5])
    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        feature_data['y_pred_xgb'] = pd.Series([float('nan')] * len(X_test))

    prophet_data = feature_data[['timestamp']].rename(columns={'timestamp': 'ds'})
    try:
        prophet_forecast = prophet_model.predict(prophet_data)
        feature_data['y_pred_prophet'] = prophet_forecast['yhat'].values
        print("Prophet predictions:", feature_data['y_pred_prophet'][:5])
    except Exception as e:
        print(f"Error in Prophet prediction: {e}")
        feature_data['y_pred_prophet'] = pd.Series([float('nan')] * len(feature_data))

    print("Predictions made for all models.")
    return feature_data, y_true

# Evaluate model bias and log results to MLflow
# Evaluate model bias and log results to MLflow
def evaluate_model_bias(feature_data, y_true):
    slicing_features = ['hour', 'day_of_week', 'month', 'season']
    model_columns = ['y_pred_rf', 'y_pred_xgb', 'y_pred_prophet']
    overall_bias_results = []
    experiment_mapping = {
    'y_pred_rf': "Random Forest Bias Evaluation",
    'y_pred_xgb': "XGBoost Bias Evaluation",
    'y_pred_prophet': "Prophet Bias Evaluation"
}

# Loop through each model and set the experiment accordingly
    for model_col in model_columns:
        # Set the experiment for the current model
        experiment_name = experiment_mapping[model_col]
        mlflow.set_experiment(experiment_name)
        
        # Start the experiment run for the model
        with mlflow.start_run(run_name=f"{model_col}_Bias Evaluation"):
            print(f"\nEvaluating bias and metrics for model: {model_col}")
            mlflow.log_param("model", model_col)

            # Loop through each slicing feature within the experiment
            for feature in slicing_features:
                if feature not in feature_data.columns:
                    print(f"Skipping feature '{feature}' as it is not present.")
                    continue

                # Create a new run for each feature within the model experiment
                with mlflow.start_run(run_name=f"{model_col}_{feature}_Feature", nested=True):
                    print(f"\nEvaluating feature: {feature} for model: {model_col}")
                    
                    # Collect metrics for each slice of the feature
                    grouped_results = []
                    for name, group in feature_data.groupby(feature):
                        # Calculate metrics
                        mae = mean_absolute_error(group['pm25'], group[model_col])
                        rmse = mean_squared_error(group['pm25'], group[model_col], squared=False)
                        r2 = r2_score(group['pm25'], group[model_col])
                        mbe = (group[model_col] - group['pm25']).mean()

                        # Log individual slice metrics
                        mlflow.log_metric(f"{feature}_{name}_MAE", mae)
                        mlflow.log_metric(f"{feature}_{name}_RMSE", rmse)
                        mlflow.log_metric(f"{feature}_{name}_R2", r2)
                        mlflow.log_metric(f"{feature}_{name}_MBE", mbe)

                        slice_metrics = {
                            "Model": model_col,
                            "Feature": feature,
                            "Slice": name,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R²": r2,
                            "MBE": mbe
                        }
                        grouped_results.append(slice_metrics)

                    # Create a DataFrame for the collected metrics
                    metrics_df = pd.DataFrame(grouped_results)

                    # Calculate average metrics for the feature
                    avg_mae = metrics_df["MAE"].mean()
                    avg_rmse = metrics_df["RMSE"].mean()
                    avg_r2 = metrics_df["R²"].mean()
                    avg_mbe = metrics_df["MBE"].mean()

                    # Log the average metrics
                    mlflow.log_metric(f"{feature}_{experiment_name}_Avg_MAE", avg_mae)
                    mlflow.log_metric(f"{feature}_{experiment_name}_Avg_RMSE", avg_rmse)
                    mlflow.log_metric(f"{feature}_{experiment_name}_Avg_R2", avg_r2)
                    mlflow.log_metric(f"{feature}_{experiment_name}_Avg_MBE", avg_mbe)

                    # Calculate deviations and bias flags
                    metrics_df['MAE Deviation'] = abs(metrics_df['MAE'] - avg_mae) / avg_mae
                    metrics_df['RMSE Deviation'] = abs(metrics_df['RMSE'] - avg_rmse) / avg_rmse
                    metrics_df['R² Deviation'] = abs(metrics_df['R²'] - avg_r2) / abs(avg_r2)
                    metrics_df['MBE Deviation'] = abs(metrics_df['MBE'] - avg_mbe) / abs(avg_mbe)

                    metrics_df['Biased MAE'] = metrics_df['MAE Deviation'] > 1
                    metrics_df['Biased RMSE'] = metrics_df['RMSE Deviation'] > 1
                    metrics_df['Biased R²'] = metrics_df['R² Deviation'] > 1
                    metrics_df['Biased MBE'] = metrics_df['MBE Deviation'] > 1

                    # Log counts of biased slices for each metric
                    biased_counts = {
                        "Biased MAE Count": metrics_df['Biased MAE'].sum(),
                        "Biased RMSE Count": metrics_df['Biased RMSE'].sum(),
                        "Biased R² Count": metrics_df['Biased R²'].sum(),
                        "Biased MBE Count": metrics_df['Biased MBE'].sum(),
                    }
                    for key, value in biased_counts.items():
                        mlflow.log_metric(key, value)

                    print(f"Logged metrics and bias flags for model {model_col}, feature {feature}")

                    for metric in ["MAE", "RMSE", "R²", "MBE"]:
                        plot_metric_distribution(metrics_df, model_col, feature, metric)


                    # Append to overall results for final analysis if needed
                    overall_bias_results.append(metrics_df)

    # Combine all bias results into a final DataFrame for reference
    bias_results_df = pd.concat(overall_bias_results, ignore_index=True)

    return bias_results_df

def plot_metric_distribution(metrics_df, model_name, feature, metric_name):
    storage_client = storage.Client()
    bucket_name = "airquality-mlops-rg"
    bucket = storage_client.bucket(bucket_name)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Slice', y=metric_name, data=metrics_df)
    plt.title(f'{model_name} - {feature} - {metric_name} Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    artifact_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    curr_dir = os.getcwd()
    # Save the plot image locally
    img_path = os.path.join(curr_dir, f'artifacts/{model_name}_{feature}_{metric_name}.png')
    plt.savefig(img_path)
    plt.close()
    print(f"Saved {metric_name} distribution plot for {model_name} - {feature} at {img_path}")
    destination_blob_name = f'artifacts/{model_name}_{feature}_{metric_name}.png'
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(img_path)
    return img_path

# Main pipeline function
def main_pipeline():
    feature_data = load_feature_data()
    rf_model, xgb_model, prophet_model = load_models()
    feature_data, y_true = make_predictions(feature_data, rf_model, xgb_model, prophet_model)
    bias_results = evaluate_model_bias(feature_data, y_true)
    print("\nModel Bias Analysis Complete.")
    return bias_results

if __name__ == "__main__":
    main_pipeline()
