
# import mlflow
# import os
# import pandas as pd
# import pickle
# import numpy as np
# import xgboost as xgb
# from dags.ModelDevelopment.Validation.XGBoost import XGBoostPM25Model
# from dags.ModelDevelopment.Validation.Prophet import ProphetPM25Model
# from dags.ModelDevelopment.Validation.RandomForest import RandomForestPM25Model

# def load_randomforest_model(filepath):
#     """Loads the Random Forest model from the given filepath."""
#     randomforest = RandomForestPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
#     model_save_path = filepath
#     with open(model_save_path, 'rb') as f:
#        randomforest = pd.read_pickle(f)
#     return randomforest

# def load_xgboost_model(filepath):
#     """Loads the XGBoost model from the given filepath."""
#     xgbmodel = xgb.XGBRegressor(random_state=42)
#     xgbmodel.load_model(filepath)
#     return xgbmodel

# def load_prophet_model(filepath):
#     """Loads the Prophet model from the given filepath."""
#     prophetmodel = ProphetPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
#     with open(filepath, 'rb') as f:
#             prophetmodel= pickle.load(f)
#     return prophetmodel

# def find_best_model_run(experiment_name):
#     """Finds the best model run in an experiment based on the lowest RMSE metric."""
#     experiment = mlflow.get_experiment_by_name(experiment_name)
#     if experiment is None:
#         print(f"Experiment '{experiment_name}' not found.")
#         return None, None, None
    
#     experiment_id = experiment.experiment_id
#     runs = mlflow.search_runs(experiment_ids=[experiment_id])
#     if runs.empty:
#         print(f"No runs found for experiment '{experiment_name}'.")
#         return None, None, None
    
#     # Identify the run with the lowest RMSE
#     best_run = runs.loc[runs['metrics.RMSE'].idxmin()]
#     best_rmse = best_run['metrics.RMSE']
#     best_run_id = best_run['run_id']
    
#     return best_run_id, best_rmse, experiment_name

# def get_model_uri_and_loader(experiment_name, best_run_id):
#     """Returns the model URI and corresponding loader function based on experiment name."""
#     if experiment_name == "PM2.5 Prophet":
#         model_uri = f"runs:/{best_run_id}/prophet_pm25_model.pth"
#         model_loader = load_prophet_model
#     elif experiment_name == "PM2.5 Random Forest":
#         model_uri = f"runs:/{best_run_id}/randomforest_pm25_model.pth"
#         model_loader = load_randomforest_model
#     elif experiment_name == "PM2.5 XGBoost Prediction":
#         model_uri = f"runs:/{best_run_id}/xgboost_pm25_model.pth"
#         model_loader = load_xgboost_model
#     else:
#         raise ValueError(f"Unsupported experiment type: {experiment_name}")
    
#     return model_uri, model_loader

# def download_and_load_model(model_uri, model_loader):
#     """Downloads the model artifact from MLflow and loads it using the provided loader function."""
#     model_path = mlflow.artifacts.download_artifacts(model_uri)
#     model = model_loader(model_path)
#     return model

# def get_best_model_and_load_weights(experiment_names):
#     """Finds and loads the best model across multiple experiments based on RMSE."""
#     best_rmse = float('inf')
#     best_model = None
#     best_experiment_name = None
#     best_run_id = None
#     for experiment_name in experiment_names:
#         run_id, rmse, exp_name = find_best_model_run(experiment_name)
#         if run_id is not None and rmse < best_rmse:
#             best_rmse = rmse
#             best_experiment_name = exp_name
#             best_run_id = run_id
#     if best_run_id:
#         model_uri, model_loader = get_model_uri_and_loader(best_experiment_name, best_run_id)
#         best_model = download_and_load_model(model_uri, model_loader)
#         print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
#         print(f"Validation RMSE: {best_rmse}")
#     else:
#         print("No valid models found across experiments.")
#     return best_model, best_rmse, best_experiment_name, best_run_id
# def main():
#     experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
#     model, best_rmse, best_experiment_name, best_run_id = get_best_model_and_load_weights(experiment_names)
#     if model:
#         print(f"Best Experiment: {best_experiment_name}")
#         print(f"Best Run ID: {best_run_id}")
#         print(f"Best RMSE: {best_rmse}")
#     else:
#         print("No model could be loaded.")

# if __name__ == "__main__":
#     main()

# # import mlflow
# # import os
# # import pandas as pd
# # import pickle
# # import numpy as np
# # import xgboost as xgb
# # from dags.ModelDevelopment.Validation.XGBoost import XGBoostPM25Model
# # from dags.ModelDevelopment.Validation.Prophet import ProphetPM25Model
# # from dags.ModelDevelopment.Validation.RandomForest import RandomForestPM25Model

# # # Model Loading Functions
# # def load_randomforest_model(filepath):
# #     """Loads the Random Forest model from the given filepath."""
# #     randomforest = RandomForestPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
# #     model_save_path = filepath
# #     with open(model_save_path, 'rb') as f:
# #        randomforest = pd.read_pickle(f)
# #     return randomforest

# # def load_xgboost_model(filepath):
# #     """Loads the XGBoost model from the given filepath."""
# #     xgbmodel = xgb.XGBRegressor(random_state=42)
# #     xgbmodel.load_model(filepath)
# #     return xgbmodel

# # def load_prophet_model(filepath):
# #     """Loads the Prophet model from the given filepath."""
# #     prophetmodel = ProphetPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
# #     with open(filepath, 'rb') as f:
# #             prophetmodel= pickle.load(f)
# #     return prophetmodel

# # # Function to Identify Best Model
# # def find_best_model_run(experiment_name):
# #     """Finds the best model run in an experiment based on the lowest RMSE metric."""
# #     experiment = mlflow.get_experiment_by_name(experiment_name)
# #     if experiment is None:
# #         print(f"Experiment '{experiment_name}' not found.")
# #         return None, None, None
    
# #     experiment_id = experiment.experiment_id
# #     runs = mlflow.search_runs(experiment_ids=[experiment_id])
# #     if runs.empty:
# #         print(f"No runs found for experiment '{experiment_name}'.")
# #         return None, None, None
    
# #     # Identify the run with the lowest RMSE
# #     best_run = runs.loc[runs['metrics.RMSE'].idxmin()]
# #     best_rmse = best_run['metrics.RMSE']
# #     best_run_id = best_run['run_id']
    
# #     return best_run_id, best_rmse, experiment_name

# # # Function to Get Model URI Based on Experiment Type
# # def get_model_uri_and_loader(experiment_name, best_run_id):
# #     """Returns the model URI and corresponding loader function based on experiment name."""
# #     if experiment_name == "PM2.5 Prophet":
# #         model_uri = f"runs:/{best_run_id}/prophet_pm25_model.pth"
# #         model_loader = load_prophet_model
# #     elif experiment_name == "PM2.5 Random Forest":
# #         model_uri = f"runs:/{best_run_id}/randomforest_pm25_model.pth"
# #         model_loader = load_randomforest_model
# #     elif experiment_name == "PM2.5 XGBoost Prediction":
# #         model_uri = f"runs:/{best_run_id}/xgboost_pm25_model.pth"
# #         model_loader = load_xgboost_model
# #     else:
# #         raise ValueError(f"Unsupported experiment type: {experiment_name}")
# #     print(model_uri)
# #     return model_uri, model_loader

# # # Function to Download and Load the Model
# # def download_and_load_model(model_uri, model_loader):
# #     """Downloads the model artifact from MLflow and loads it using the provided loader function."""
# #     model_path = mlflow.artifacts.download_artifacts(model_uri)
# #     print(model_path)
# #     model = model_loader(model_path)
# #     return model

# # # Main Function to Get the Best Model Across Experiments
# # def get_best_model_and_load_weights(experiment_names):
# #     """Finds and loads the best model across multiple experiments based on RMSE."""
# #     best_rmse = float('inf')
# #     best_model = None
# #     best_experiment_name = None
# #     best_run_id = None
# #     rmse_results = {}
# #     for experiment_name in experiment_names:
# #         run_id, rmse, exp_name = find_best_model_run(experiment_name)
# #         if experiment_name == "PM2.5 Prophet":
# #             rmse_results["Prophet"] = rmse
# #         if experiment_name == "PM2.5 Random Forest":
# #             rmse_results["Random"] = rmse
# #         if experiment_name == "PM2.5 XGBoost Prediction":
# #             rmse_results["XGBoost"] = rmse
# #         if run_id is not None and rmse < best_rmse:
# #             best_rmse = rmse
# #             best_experiment_name = exp_name
# #             best_run_id = run_id
# #     print(best_experiment_name)
# #     print(best_run_id)

# #     if best_run_id:
# #         model_uri, model_loader = get_model_uri_and_loader(best_experiment_name, best_run_id)
# #         print(model_uri)
# #         best_model = download_and_load_model(model_uri, model_loader)

# #         print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
# #         print(f"Validation RMSE: {best_rmse}")
# #     else:
# #         print("No valid models found across experiments.")
    
# #     return best_model, best_rmse, best_experiment_name, best_run_id

# # def get_bias_results(experiment_names):
# #     best_metrics = {}
# #     for experiment_name in experiment_names:
# #         # Get the experiment ID
# #         experiment = mlflow.get_experiment_by_name(experiment_name)
# #         if experiment is None:
# #             print(f"Experiment '{experiment_name}' not found.")
# #             continue
# #         experiment_id = experiment.experiment_id

# #         # Query all runs in the experiment
# #         runs = mlflow.search_runs(experiment_ids=[experiment_id])

# #         # Ensure there are runs to process
# #         if runs.empty:
# #             print(f"No runs found for experiment '{experiment_name}'.")
# #             continue

# #         # Find the minimum and maximum values for each metric
# #         best_mae_run = runs.loc[runs['metrics.MAE'].idxmin()]
# #         best_rmse_run = runs.loc[runs['metrics.RMSE'].idxmin()]
# #         best_r2_run = runs.loc[runs['metrics.R²'].idxmax()]
# #         best_mbe_run = runs.loc[runs['metrics.MBE'].idxmin()]

# #         # Record the best values and corresponding run IDs
# #         best_metrics[experiment_name] = {
# #             "Best MAE": (best_mae_run['metrics.MAE'], best_mae_run['run_id']),
# #             "Best RMSE": (best_rmse_run['metrics.RMSE'], best_rmse_run['run_id']),
# #             "Best R²": (best_r2_run['metrics.R²'], best_r2_run['run_id']),
# #             "Best MBE": (best_mbe_run['metrics.MBE'], best_mbe_run['run_id']),
# #         }
# #     return best_metrics

# # # Entry Point Function
# # def main():
# #     experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
# #     experiment_names_2 = ["Random Forest Bias Evaluation", "XGBoost Bias Evaluation", "Prophet Bias Evaluation"]
# #     model, best_rmse, best_experiment_name, best_run_id,rmse_results  = get_best_model_and_load_weights(experiment_names)
# #     bias_results = get_bias_results(experiment_names_2)
# #     print(rmse_results)
# #     print(bias_results)
# #     if model:
# #         print(f"Best Experiment: {best_experiment_name}")
# #         print(f"Best Run ID: {best_run_id}")
# #         print(f"Best RMSE: {best_rmse}")
# #     else:
# #         print("No model could be loaded.")

# # if __name__ == "__main__":
# #     main()



# Main Function to Get the Best Model Across Experiments

import mlflow
import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from dags.ModelDevelopment.Validation.XGBoost import XGBoostPM25Model
from dags.ModelDevelopment.Validation.Prophet import ProphetPM25Model
from dags.ModelDevelopment.Validation.RandomForest import RandomForestPM25Model

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
# def get_model_uri_and_loader(experiment_name, best_run_id):
#     """Returns the model URI and corresponding loader function based on experiment name."""
#     if experiment_name == "PM2.5 Prophet":
#         model_uri = f"runs:/{best_run_id}/artifacts/prophet_pm25_model.pth"
#         model_loader = load_prophet_model
#     elif experiment_name == "PM2.5 Random Forest":
#         model_uri = f"runs:/{best_run_id}/artifacts/randomforest_pm25_model.pth"
#         model_loader = load_randomforest_model
#     elif experiment_name == "PM2.5 XGBoost Prediction":
#         model_uri = f"runs:/{best_run_id}/artifacts/xgboost_pm25_model.pth"
#         model_loader = load_xgboost_model
#     else:
#         raise ValueError(f"Unsupported experiment type: {experiment_name}")
#     return model_uri, model_loader

def get_bias_results(experiment_names):
    slicing_features = ['hour', 'day_of_week', 'month', 'season']
    best_metrics = {}

    for experiment_name in experiment_names:
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            continue
        experiment_id = experiment.experiment_id

        # Query all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Ensure there are runs to process
        if runs.empty:
            print(f"No runs found for experiment '{experiment_name}'.")
            continue

        # Initialize a dictionary for this experiment to store best metric values
        best_metrics[experiment_name] = {}

        for feature in slicing_features:
            # Find the best runs based on the specific metrics for each feature
            best_mae_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_MAE'].idxmin()]
            best_rmse_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_RMSE'].idxmin()]
            best_r2_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_R2'].idxmax()]
            best_mbe_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_MBE'].idxmin()]

            # Store only the metric values and run IDs for the best runs
            best_metrics[experiment_name][f"MAE_{feature}"] = {
                "value": best_mae_run[f"metrics.{feature}_{experiment_name}_Avg_MAE"],
                "run_id": best_mae_run["run_id"]
            }
            best_metrics[experiment_name][f"RMSE_{feature}"] = {
                "value": best_rmse_run[f"metrics.{feature}_{experiment_name}_Avg_RMSE"],
                "run_id": best_rmse_run["run_id"]
            }
            best_metrics[experiment_name][f"R2_{feature}"] = {
                "value": best_r2_run[f"metrics.{feature}_{experiment_name}_Avg_R2"],
                "run_id": best_r2_run["run_id"]
            }
            best_metrics[experiment_name][f"MBE_{feature}"] = {
                "value": best_mbe_run[f"metrics.{feature}_{experiment_name}_Avg_MBE"],
                "run_id": best_mbe_run["run_id"]
            }

    return best_metrics

# Main Function to Get the Best Model Across Experiments
def get_best_model_and_load_weights(experiment_names):
    """Finds and loads the best model across multiple experiments based on RMSE."""
    best_rmse = float('inf')
    best_model = None
    best_experiment_name = None
    best_run_id = None
    rmse_results = {}
    
    for experiment_name in experiment_names:
        run_id, rmse, exp_name = find_best_model_run(experiment_name)
        if experiment_name == "PM2.5 Prophet":
            rmse_results["Prophet"] = rmse
        if experiment_name == "PM2.5 Random Forest":
            rmse_results["Random"] = rmse
        if experiment_name == "PM2.5 XGBoost Prediction":
            rmse_results["XGBoost"] = rmse

        if run_id is not None and rmse < best_rmse:
            best_rmse = rmse
            best_experiment_name = exp_name
            best_run_id = run_id

    if best_run_id:
        #model_uri, model_loader = get_model_uri_and_loader(best_experiment_name, best_run_id)
        if best_experiment_name == "PM2.5 Prophet":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"dags/weights/prophet_pm25_model.pth")
            best_model = load_prophet_model(directory_weights_path)
        if best_experiment_name == "PM2.5 Random Forest":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"dags/weights/randomforest_pm25_model.pth")
            best_model = load_randomforest_model(directory_weights_path)
        if best_experiment_name == "PM2.5 XGBoost Prediction":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"dags/weights/xgboost_pm25_model.pth")
            best_model = load_xgboost_model(directory_weights_path)

        print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
        print(f"Validation RMSE: {best_rmse}")
    else:
        print("No valid models found across experiments.")
    
    return best_model, best_rmse, best_experiment_name, best_run_id, rmse_results


# Define a penalty for bias metrics (lower values preferred)
def compute_bias_score(bias_metrics, metric_weights):
    score = 0
    for feature, metrics in bias_metrics.items():
        for metric, details in metrics.items():
            metric_name = metric.split('_')[0]  # Extract MAE, RMSE, etc.
            if metric_name in metric_weights:
                score += metric_weights[metric_name] * abs(details["value"])
    return score

# Main function to select the best model based on RMSE and bias scores
def select_best_model(rmse_results, bias_results, metric_weights):
    best_model = None
    best_combined_score = float('inf')
    
    for model, rmse in rmse_results.items():
        # Skip models without RMSE values
        if rmse is None:
            continue
        
        # Calculate bias score if bias results are available
        bias_key = f"{model} Bias Evaluation"
        if bias_key in bias_results:
            bias_score = compute_bias_score(bias_results[bias_key], metric_weights)
        else:
            bias_score = float('inf')  # Penalize models without bias metrics

        # Compute combined score as weighted RMSE + Bias Score
        combined_score = metric_weights["RMSE"] * rmse + bias_score
        print(f"{model} - RMSE: {rmse}, Bias Score: {bias_score}, Combined Score: {combined_score}")
        
        # Update best model if the combined score is lower
        if combined_score < best_combined_score:
            best_model = model
            best_combined_score = combined_score

    print(f"\nBest model selected: {best_model} with combined score: {best_combined_score}")
    return best_model, best_combined_score




# Entry Point Function
def main():
    experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
    experiment_names_2 = ["Random Forest Bias Evaluation", "XGBoost Bias Evaluation", "Prophet Bias Evaluation"]
    model, best_rmse, best_experiment_name, best_run_id, rmse_results= get_best_model_and_load_weights(experiment_names)
    bias_results = get_bias_results(experiment_names_2)
    # Define weights for each metric - adjust these based on importance
    metric_weights = {
        "RMSE": 0.5,        # Overall importance of RMSE
        "MAE": 0.2,         # Mean Absolute Error weight
        "R2": 0.2,          # R-squared weight
        "MBE": 0.1          # Mean Bias Error weight
    }
    # Run the selection function
    best_model, best_combined_score = select_best_model(rmse_results, bias_results, metric_weights)
    if best_model == "Prophet":
        curr_dir = os.getcwd()
        directory_weights_path = os.path.join(curr_dir,"dags/weights/prophet_pm25_model.pth")
        model = load_prophet_model(directory_weights_path)
    if best_model == "Random":
        curr_dir = os.getcwd()
        directory_weights_path = os.path.join(curr_dir,"dags/weights/randomforest_pm25_model.pth")
        model = load_randomforest_model(directory_weights_path)
    if best_model == "XGBoost":
        curr_dir = os.getcwd()
        directory_weights_path = os.path.join(curr_dir,"dags/weights/xgboost_pm25_model.pth")
        model = load_xgboost_model(directory_weights_path)
    print(best_model)
    print(best_combined_score)
    print(bias_results)
    print(rmse_results)
    if model:
        print(f"Best Experiment: {best_experiment_name}")
        print(f"Best Run ID: {best_run_id}")
        print(f"Best RMSE: {best_rmse}")
    else:
        print("No model could be loaded.")

if __name__ == "__main__":
    main()
