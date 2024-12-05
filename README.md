# Air-Quality
This repository contains the code and resources for predicting PM2.5 levels for location Miami-Fort Lauderdale-Miami Beach
from 2022 Jan to 2023 Dec using air quality data from OpenAQ. It includes data collection, preprocessing, and modeling scripts along with analysis and visualizations. The project aims to forecast PM2.5 levels and provide insights into air pollution trends.

## pre-requisites

git

python>=3.8

docker desktop is running

DVC

        
#### Installation of Docker desktop For macOS:

##### Download Docker Desktop
```bash
curl -O https://desktop.docker.com/mac/stable/Docker.dmg
```
##### Mount the DMG File
```bash
sudo hdiutil attach Docker.dmg
 ```
##### Copy Docker to Applications
```bash
sudo /Volumes/Docker/Docker.app/Contents/MacOS/install
```
##### Eject the DMG
```bash
sudo hdiutil detach /Volumes/Docker
```
##### Start Docker Desktop
```bash
open/Applications/Docker.app
```
##### Verify Installation
```bash
docker --version
```
 
#### Installation of Docker desktop For Windows:
##### Download Docker Desktop:  (Execute in Powershell) 

```powershell
Invoke-WebRequest -Uri "https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe" -OutFile "$env:USERPROFILE\Downloads\DockerDesktopInstaller.exe"
```
##### Install Docker Desktop Silently:
 ```powershell 
 Start-Process -FilePath "$env:USERPROFILE\Downloads\DockerDesktopInstaller.exe" -ArgumentList "install", "--quiet", "--accept-license" -Wait
 ```          
##### Verify Installation:
 ```powershell            
 docker --version
```

## Steps to Execute Data Pipeline 
 
#### Step 1: Clone the Repository 
```bash
git clone https://github.com/KARSarma/Air-Quality.git
```
#### Step 2: Navigate to directory
```bash
cd Air-Quality
```
#### Step 3: Initialize DVC for Data Version Control
```bash  
dvc pull
```
#### Step 4: Initialize the Airflow Database
Make sure to keep the docker desktop up and running
```bash
docker compose up airflow-init
```
#### Step 5: Start the Airflow Services
```bash
docker compose up
```
#### Step 6: Access the Airflow Web UI
Navigate to http://localhost:8080

Log in using the default credentials:

Username: airflow

Password: airflow

#### Step 7: Trigger the Dag

In the Airflow UI, locate the dag_script to run.

Toggle the DAG to “On” (enable it) if it is not already active.

Click on the DAG name to open its details page and manually trigger a run by clicking the “Trigger DAG” button.

![Alt text](https://github.com/KARSarma/Air-Quality/blob/81e34ebcc5a4bc26fd3b075f19f1a76f901b983e/Trigger_dag.png)

#### Step 8: Stopping the Airflow Services
``` bash
docker compose down
```
       
### Folder Structure Overview

```plaintext
Air-Quality/
-DataPreprocessing
│
├── src
│   │   ├── Schema
│   │   │   ├── check_schema_original_airpollution.py
│   │   │   ├── test_schema
│   │   │   │   └── check_output_data_schema.py
│   │   │   └── train_schema
│   │   │       └── check_output_data_schema.py
│   │   ├── data_air.py
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── data_store_pkl_files
│   │   │   ├── air_pollution.pkl
│   │   │   ├── csv
│   │   │   ├── test_data
│   │   │   │   ├── cleaned_test_data.pkl
│   │   │   │   ├── feature_eng_test_data.pkl
│   │   │   │   ├── no_anamoly_test_data.pkl
│   │   │   │   ├── no_null_test_data.pkl
│   │   │   │   ├── pivoted_test_data.pkl
│   │   │   │   └── test_data.pkl
│   │   │   └── train_data
│   │   │       ├── cleaned_train_data.pkl
│   │   │       ├── feature_eng_train_data.pkl
│   │   │       ├── no_anamoly_train_data.pkl
│   │   │       ├── no_null_train_data.pkl
│   │   │       ├── pivoted_train_data.pkl
│   │   │       └── train_data.pkl
│   │   ├── test
│   │   │   └── data_preprocessing
│   │   │       ├── __init__.py
│   │   │       ├── anamoly_detection.py
│   │   │       ├── check_missing_values.py
│   │   │       ├── feature_eng.py
│   │   │       ├── pivoting_data.py
│   │   │       └── removal_of_uneccesary_cols.py
│   │   └── train
│   │       └── data_preprocessing
│   │           ├── __init__.py
│   │           ├── anamoly_detection.py
│   │           ├── check_missing_values.py
│   │           ├── feature_eng.py
│   │           ├── pivoting_data.py
│   │           └── removal_of_uneccesary_cols.py
│   └── test
│       ├── __init__.py
│       ├── test_data
│       │   ├── __init__.py
│       │   ├── test_anamoly_detection.py
│       │   ├── test_check_missing_values.py
│       │   ├── test_feature_eng.py
│       │   ├── test_pivoting_data.py
│       │   └── test_removal_of_uneccesary_cols.py
│       └── train_data
│           ├── __init__.py
│           ├── test_anamoly_detection.py
│           ├── test_check_missing_values.py
│           ├── test_feature_eng.py
│           ├── test_pivoting_data.py
│           └── test_removal_of_uneccesary_cols.py
├── dags/
│   ├── DataPreprocessing/
│   │   ├── src/
│   │   │   ├── Schema/
│   │   │   ├── data_store_pkl_files/
│   │   │   │   ├── test_data/
│   │   │   │   └── train_data/
│   │   ├── test/
│   │   └── train/
│   ├── __init__.py
│   ├── air_pollution_stats.json
│   ├── custom_schema_generated_from_api.json
│   └── dag_script.py
├── docker-compose.yaml
├── ModelDevelopment/
└── README.md

```
## Detailed Breakdown

dags/:                            Contains all Airflow DAGs and scripts for orchestrating the data processing pipeline.

DataPreprocessing/:               Houses the preprocessing modules and schemas.

src/:                             Source directory for data loading and schema validation.

Schema/:                          Directory containing schema validation scripts and JSON schema files.

data_store_pkl_files/:            Directory for serialized data files, raw csv files organized by:

test_data/:                       Processed test dataset files.

train_data/:                      Processed training dataset files.

test/ & train/:                   Subdirectories for test and training preprocessing scripts.

air_pollution_stats.json:         JSON file with statistical summaries of air pollution data.

custom_schema_generated_from_api.json: JSON schema from API data, ensuring consistent structure.

dag_script.py:                    The main Airflow DAG script for managing task sequences and scheduling.

docker-compose.yaml:              Docker Compose configuration file to set up Airflow and related services.

ModelDeployment/, ModelDevelopment/, ModelLogging/, ModelMonitoring/: Directories for managing different stages of the model lifecycle, from development to deployment.

test/:                            Contains test cases required to run after datapipeline using GitHub-Actions

README.md:                        Project documentation file.


## DVC

DVC is configured when data file is loaded initially and after final preprocessing in the gcs bucket configured remotely on GCP cloud and when modifications are executed locally the changes will be reflected in gcp cloud after dvc push. Latest file need to be fetched before executing data pipeline in the cloud composer. For now, we have configured dvc on cloud but data pipeline is executed in airflow locally, will be used once we setup the cloud composer data pipeline. 

## Data Pipeline Steps

The data pipeline for the Air-Quality project consists of a series of steps to preprocess, transform, and model air quality data. Each stage of the pipeline performs a specific function that prepares the data for PM2.5  predictions.
 
 ### Pipeline Steps

#### 1. Data Collection

This step collects raw air quality data from OpenAQ API and stores it locally. The script is designed to fetch historical PM2.5 readings based on specified parameters, such as geographic region and time range. Also stacks csv files

 Scripts: 

 
  dags/DataPreprocessing/src/data_loader.py
  
  dags/DataPreprocessing/src/data_air.py

#### 2. Data Bias

Analyzes potential biases in the data that could impact model predictions.

Script:
 
dags/DataPreprocessing/src/data_bias_check_final.py

#### 3.Schema Validation

Ensures that the collected data adheres to a predefined schema. This validation step is crucial to check for consistency and correctness in the dataset structure. The schema includes necessary columns and data types for downstream processing. It is verified after loading the initial data and after the final feature engineering.

Script: 

dags/DataPreprocessing/src/Schema/check_schema_original_airpollution.py

dags/DataPreprocessing/src/Schema/test_schema/check_output_data_schema.py

dags/DataPreprocessing/src/Schema/train_schema/check_output_data_schema.py


#### 4.Data Preprocessing

In this step, the data is cleaned and prepared for analysis. It includes handling missing values, detecting anomalies, and performing initial feature engineering. The preprocessed data is stored as .pkl files in data_store_pkl_files for both training and testing sets.

Scripts:

 dags/DataPreprocessing/src/data_split.py
 
 dags/DataPreprocessing/train/data_preprocessing/pivoting_data.py
 
 dags/DataPreprocessing/train/data_preprocessing/removal_of_uneccesary_cols.py
 
 dags/DataPreprocessing/train/data_preprocessing/check_missing_values.py
 
 dags/DataPreprocessing/train/data_preprocessing/anamoly_detection.py
            

#### 5.Feature Engineering

This feature engineering step enhances model performance by capturing temporal patterns and environmental influences essential for air quality prediction. Techniques applied include lag and lead features to account for past and immediate future pollutant levels, rolling statistics (mean, sum, min, max) to summarize recent trends, and differencing to highlight rate of change. Cosine similarity with historical patterns is used to identify recurring pollution trends, while time-based features (hour, day, month) help capture cyclical variations providing a robust set of features for effective PM2.5 prediction.

Script:

dags/DataPreprocessing/train/data_preprocessing/feature_eng.py

#### 6. Data Validation

Runs validation checks to ensure the test,train data meets quality standards before entering the modeling phase. This includes confirming the absence of missing values and verifying data transformations. 

Script: 
      
dags/DataPreprocessing/test/data_preprocessing/check_output_data_schema.py

![Alt text](https://github.com/KARSarma/Air-Quality/blob/979abab65dd00fbfdd0d398044b2ecc5b7ccbc70/Airflow_dags.png)

The above image shows the entire airflow dag data pipeline workflow.

## Email-Notification

Email configurations are also included to update about the success and failure of the dags through smtp. Below mentioned email is used as part of configuration and used same for triggering anomoly alerts. SMTP setup is configured in docker compose file, app code is generated accordingly and used as passcode. Currently recieving and sending mail is same which is mentioned below

Email: anirudhak881@gmail.com

## Model Development

Code for initiating the model development pipeline locally can be done using docker as follows

After cloning the repo as given above open the model development folder
```powershell
cd ModelDevelopment
```

Once you are inside the model development folder run these docker commands to run the model development pipeline 

```powershell
docker build -t <name_of_container> .
docker run <name_of_container>
```
### Folder Structure Overview

```plaintext
Air-Quality
└── ModelDevelopment
   -DataPreprocessing
        │
        ├── src
        │   │   ├── Schema
        │   │   │   ├── check_schema_original_airpollution.py
        │   │   │   ├── test_schema
        │   │   │   │   └── check_output_data_schema.py
        │   │   │   └── train_schema
        │   │   │       └── check_output_data_schema.py
        │   │   ├── data_air.py
        │   │   ├── data_loader.py
        │   │   ├── data_split.py
        │   │   ├── data_store_pkl_files
        │   │   │   ├── air_pollution.pkl
        │   │   │   ├── csv
        │   │   │   ├── test_data
        │   │   │   │   ├── cleaned_test_data.pkl
        │   │   │   │   ├── feature_eng_test_data.pkl
        │   │   │   │   ├── no_anamoly_test_data.pkl
        │   │   │   │   ├── no_null_test_data.pkl
        │   │   │   │   ├── pivoted_test_data.pkl
        │   │   │   │   └── test_data.pkl
        │   │   │   └── train_data
        │   │   │       ├── cleaned_train_data.pkl
        │   │   │       ├── feature_eng_train_data.pkl
        │   │   │       ├── no_anamoly_train_data.pkl
        │   │   │       ├── no_null_train_data.pkl
        │   │   │       ├── pivoted_train_data.pkl
        │   │   │       └── train_data.pkl
        │   │   ├── test
        │   │   │   └── data_preprocessing
        │   │   │       ├── __init__.py
        │   │   │       ├── anamoly_detection.py
        │   │   │       ├── check_missing_values.py
        │   │   │       ├── feature_eng.py
        │   │   │       ├── pivoting_data.py
        │   │   │       └── removal_of_uneccesary_cols.py
        │   │   └── train
        │   │       └── data_preprocessing
        │   │           ├── __init__.py
        │   │           ├── anamoly_detection.py
        │   │           ├── check_missing_values.py
        │   │           ├── feature_eng.py
        │   │           ├── pivoting_data.py
        │   │           └── removal_of_uneccesary_cols.py
            ├── dags/
                └── artifacts/
                    ├── learning_rate_sensitivity_xgboost.png
                    ├── max_depth_sensitivity_xgboost.png
                    ├── n_estimators_sensitivity_randomForest.png
                    ├── n_estimators_sensitivity_xgboost.png
                    ├── pm25_actual_vs_predicted_Prophet.png
                    ├── pm25_actual_vs_predicted_RandomForest.png
                    ├── pm25_actual_vs_predicted_XGBoost.png
                    ├── shap_summary_plot_prophet.png
                    ├── shap_summary_plot_randomforest.png
                    ├── shap_summary_plot_xgboost.png
                    ├── y_pred_prophet_day_of_week_MAE.png
                    ├── y_pred_prophet_day_of_week_MBE.png
                    ├── y_pred_prophet_day_of_week_RMSE.png
                    ├── y_pred_prophet_day_of_week_R².png
                    ├── y_pred_prophet_hour_MAE.png
                    ├── y_pred_prophet_hour_MBE.png
                    ├── y_pred_prophet_hour_RMSE.png
                    ├── y_pred_prophet_hour_R².png
                    ├── y_pred_prophet_month_MAE.png
                    ├── y_pred_prophet_month_MBE.png
                    ├── y_pred_prophet_month_RMSE.png
                    ├── y_pred_prophet_month_R².png
                    ├── y_pred_prophet_season_MAE.png
                    ├── y_pred_prophet_season_MBE.png
                    ├── y_pred_prophet_season_RMSE.png
                    ├── y_pred_prophet_season_R².png
                    ├── y_pred_rf_day_of_week_MAE.png
                    ├── y_pred_rf_day_of_week_MBE.png
                    ├── y_pred_rf_day_of_week_RMSE.png
                    ├── y_pred_rf_day_of_week_R².png
                    ├── y_pred_rf_hour_MAE.png
                    ├── y_pred_rf_hour_MBE.png
                    ├── y_pred_rf_hour_RMSE.png
                    ├── y_pred_rf_hour_R².png
                    ├── y_pred_rf_month_MAE.png
                    ├── y_pred_rf_month_MBE.png
                    ├── y_pred_rf_month_RMSE.png
                    ├── y_pred_rf_month_R².png
                    ├── y_pred_rf_season_MAE.png
                    ├── y_pred_rf_season_MBE.png
                    ├── y_pred_rf_season_RMSE.png
                    ├── y_pred_rf_season_R².png
                    ├── y_pred_xgb_day_of_week_MAE.png
                    ├── y_pred_xgb_day_of_week_MBE.png
                    ├── y_pred_xgb_day_of_week_RMSE.png
                    ├── y_pred_xgb_day_of_week_R².png
                    ├── y_pred_xgb_hour_MAE.png
                    ├── y_pred_xgb_hour_MBE.png
                    ├── y_pred_xgb_hour_RMSE.png
                    ├── y_pred_xgb_hour_R².png
                    ├── y_pred_xgb_month_MAE.png
                    ├── y_pred_xgb_month_MBE.png
                    ├── y_pred_xgb_month_RMSE.png
                    ├── y_pred_xgb_month_R².png
                    ├── y_pred_xgb_season_MAE.png
                    ├── y_pred_xgb_season_MBE.png
                    ├── y_pred_xgb_season_RMSE.png
                    ├── y_pred_xgb_season_R².png

            ├── __init__.py
    ├── ModelBias
    │   └── Model_bias.py
    ├── Training
    │   ├── Prophet.py
    │   ├── RandomForest.py
    │   └── XGBoost.py
    ├── Validation
    │   ├── Prophet.py
    │   ├── RandomForest.py
    │   └── XGBoost.py
    ├── Dockerfile
    ├── bestmodel.py
    └── requirements.txt
└── docker-compose.yaml  
```

### Model Development scripts

**```ModelBias/Model_bias.py```**: This script contains the removal of bias from the model.

**```Training/Prophet.py```**: This script contains the model run of Prophet regression model on training data.

**```Training/RandomForest.py```**: This script contains the model run of Random forest regression model on training data.

**```Training/XGBoost.py```**: This script contains the model run of XGBoost regression model on training data.

**```Validation/Prophet.py```**: This script contains the model run of Prophet regression model on validation data.

**```Validation/RandomForest.py```**: This script contains the model run of Random forest regression model on validation data.

**```Validation/XGBoost.py```**: This script contains the model run of XGBoost regression model on validation data.

**```Dockerfile```**: This Dockerfile sets up a Python 3.8 environment for a machine learning project, configuring the container’s working directory as `/app`. It installs dependencies from `requirements.txt` and copies the source code into the container. Additionally, it sets up MLflow for tracking experiments and custom environment paths for model development.

**```Bestmodel.py```**: The `bestmodel.py` script is designed to identify and load the best-performing model from multiple experiments, likely focused on air quality predictions. It imports various machine learning libraries and models (e.g., XGBoost, Prophet, RandomForest) and defines a wrapper class for model predictions. This setup allows standardized input handling and prediction output across different model types using MLflow's model packaging framework.

**```requirements.txt```**: The `requirements.txt` file lists dependencies for a machine learning project, including essential libraries like `numpy`, `pandas`, and `scikit-learn` for data manipulation and modeling. It includes `mlflow` for experiment tracking, `xgboost` for advanced machine learning algorithms, and `prophet` for time series forecasting. The list suggests this environment is configured for data analysis, model building, and cloud storage integration with `google-cloud-storage`.



## Model Development and ML Code

### Training

#### Prophet.py
**Data Loading and Preprocessing**: Loads and preprocesses the data (PM2.5 measurements).

**Model Training**: A Prophet model is trained on the processed data.

**Model Wrapping**: The trained Prophet model is wrapped to be logged with MLflow.

**Model Logging**: MLflow logs the model, metrics (like training duration), and artifacts (like the trained model weights)

#### RandomForest.py
**Data Loading and Preprocessing**: The script loads training data, extracts Box-Cox transformed PM2.5 values as the target variable, and prepares the feature set.

**Hyperparameter Tuning with Grid Search**: It uses GridSearchCV to tune the model's hyperparameters (specifically, the number of trees in the forest) and selects the best configuration based on cross-validation with negative mean squared error as the scoring metric.

**Model Training and Logging**: The model is trained using the best hyperparameters, and the training process is logged using MLflow. The model and training duration are logged as metrics and artifacts in MLflow for tracking.

**Model Saving**: After training, the model is saved to disk and logged as an artifact in MLflow for later use.

#### XGBoost.py
**Hyperparameter Tuning**: A grid search over hyperparameters (n_estimators, learning_rate, etc.) is performed to find the best model.

**Model Training**: The model is trained using the selected parameters.

**Logging and Saving**: The model and its performance metrics (like training time) are logged using MLflow, and the trained model is saved to disk.

**Experiment Tracking**: The entire workflow is tracked in MLflow, allowing for reproducibility and comparison of experiments.

### Validation

#### Prophet.py
This script loads training and test data, trains and evaluates a Prophet model on PM2.5 values, performs SHAP analysis for interpretability, logs metrics, parameters, and artifacts to MLflow, and visualizes the actual vs. predicted results

#### RandomForest.py
This script loads training and test data, trains and evaluates a Random Forest model on PM2.5 values, performs SHAP analysis for interpretability, runs hyperparameter sensitivity analysis to assess the model's response to key hyperparameters, logs metrics, parameters, and artifacts to MLflow, and visualizes actual vs. predicted results.

#### XGBoost.py
This script loads training and test data, trains and evaluates an XGBoost model on PM2.5 values, performs SHAP analysis for interpretability, conducts hyperparameter sensitivity analysis on key parameters, logs metrics, parameters, and artifacts to MLflow, and visualizes the actual vs. predicted results.

### ModelBias

#### Model_bias.py
Loads feature-engineered data and three trained models.

Generates predictions for each model, then calculates metrics (MAE, RMSE, R², MBE) across feature-based slices, like hour, day_of_week, and season.

Evaluates bias by comparing metric deviations within each slice to detect performance biases.

Logs results to MLflow, creating an organized record of model performance under different feature slices.

### bestmodel.py
**Model Evaluation**:
The script starts by identifying the best models in different experiments based on their RMSE and bias metrics.

**Combining RMSE and Bias Metrics**:
Both RMSE (a measure of prediction error) and bias metrics (such as MAE, MBE, and R2) are considered in model selection, allowing a comprehensive evaluation of model performance.

**Rollback Mechanism**:
Before registering a new model, the script checks if any model already registered in MLflow has a better or equal RMSE. If so, the script skips registering the new model, ensuring that only models with improvements are pushed to the registry.

**Model Registration**:
If no better model exists in the registry, the script logs the selected model (based on its combined RMSE and bias score) to MLflow’s model registry.

### Dockerfile

**Base Image**: Uses an official Python 3.8 image to set up the environment for the container.

**Set Working Directory**: Sets /app as the working directory for subsequent commands.

**Install Dependencies**: Installs Python dependencies listed in requirements.txt via pip.

**Copy Source Code**: Copies the entire project directory into the container's /app folder.

**Set MLflow Tracking URI**: Configures MLflow to track experiments and store results in the container.

**Update Python Path**: Adds /app/ModelDevelopment to the Python path for easy module access.

**Create Directories for Output Files**: Creates directories /app/weights and /app/artifacts for model outputs.

**Add Execute Permissions**: Grants execute permissions to Python scripts in Training, Validation, and ModelBias folders.

**Define Command to Run Scripts Sequentially**: Specifies the order of script execution for training, validation, bias 
evaluation, and model selection.

#### Script order:
Training/Prophet.py -> Validation/Prophet.py -> Training/RandomForest.py -> Validation/RandomForest.py -> Training/XGBoost.py -> Validation/XGBoost.py -> ModelBias/Model_bias.py -> bestmodel.py


## Hyperparameter tuning


#### 1. Search Space
   
   **Random Forest**: Parameters like n_estimators, max_depth, and min_samples_split are tuned.

   **XGBoost**: Parameters such as learning_rate, max_depth, n_estimators, and subsample are explored.

   **Prophet**: Parameters like growth, changepoint_prior_scale, and seasonality_prior_scale are adjusted.

#### 2. Tuning Process
   
   **Random Forest & XGBoost**: GridSearchCV is used to perform exhaustive search with cross-validation to find the best hyperparameters.

   **Prophet**: Hyperparameters are manually selected without automated tuning.

##  Experiment Tracking and Results

**Experiment Tracking with MLflow**: 
Utilized MLflow to log experiments, track model performance, and manage hyperparameters for various models (Prophet, Random Forest, XGBoost).

**Logging Key Metrics**: 
Hyperparameters, model metrics (RMSE, MAE, R2, MBE), and model versions were tracked for each run.

**Model Comparison**: 
Visualizations and metrics from different models were compared to identify the best-performing one based on RMSE and bias.

**Model Selection**: 
The final model was selected based on the lowest RMSE and bias scores, ensuring optimal performance.

**Model Registration**: 
The best model was registered in MLflow’s model registry for version control and easy access.

**Rollback Mechanism**: 
The system checks the existing models in the registry to ensure that new models have a lower RMSE before being registered, allowing for rollback if necessary

## Model Sensitivity analysis

SHAP was utilized to interpret and explain the feature importance in predicting PM2.5 levels. By analyzing SHAP values, you can understand which features are driving the model’s predictions, how they interact, and how they influence the PM2.5 forecast. This helps ensure model transparency, improves trust, and provides actionable insights into air quality prediction

Hyperparameter Sensitivity analysis was conducted in XGBoost(params - n_estimators, learning_rate, max_depth) and Random Forest (n_estimators),Iterated over a range of values for each hyperparameter, training and evaluating the model on each setting,Calculated RMSE for each hyperparameter configuration to assess model performance, logged and saved hyperparameter settings using MLFlow,Generated a visualization showing the relationship between hyperparameter values and RMSE.

## Model Bias Detection (using Slicing Techniques)

**Perform Slicing**:
The dataset is segmented into subgroups based on slicing features like hour, day_of_week, month, and season. These slices allow for evaluating model performance across different data subgroups.

**Track Metrics Across Slices**:
Key metrics such as MAE, RMSE, R², and MBE are computed for each slice of data. These metrics are tracked and logged in MLflow to monitor performance disparities across subgroups.

**Bias Mitigation**:
Significant deviations from average metrics flag biased slices for further review.

**Document Bias Mitigation**:
Bias detection results, including biased slices and their metrics, are logged into MLflow for transparency. Visualizations are also generated to document disparities and facilitate further analysis.

## CI/CD Pipeline Automation for Model Development

Github Actions is configured using .github/workflows/docker_run_CI_CD.yml which does the following

**CI/CD Setup for Model Training**:
The pipeline is triggered on push or pull_request events to the main branch. It checks out the repository code, sets up Docker, builds the image, and runs model training scripts (Prophet, RandomForest, XGBoost) within the container.

**Automated Model Validation**:
The pipeline automatically runs validation scripts (Validation/Prophet.py, Validation/RandomForest.py, Validation/XGBoost.py) after training. It sends email notifications about validation success or failure and prevents further steps if validation fails.

**Automated Model Bias Detection**:
The pipeline runs a bias detection script (ModelBias/Model_bias.py) after model training. It sends an email notification on bias detection success or failure, allowing for alerts if any bias is detected across data slices.

**Model Deployment or Registry Push**:
when bestmodel.py is executed, from the set of experiments, it identifies the model with the lowest RMSE and evaluate it for bias metrics.
It uses a weighted combination of RMSE and bias metrics to determine the best model which is pushed to the Model Registry.

**Notifications and Alerts**:
Email notifications are sent for training completion, success or failure of individual scripts, and overall pipeline status. This ensures visibility into pipeline progress and any issues that arise.

**Rollback Mechanism**:
The bestmodel.py takes care of rollback mechanism as follows. The registry is checked for any existing model and  its RMSE compared with the new model.
Skip registration if the existing model has a better or equal RMSE.
Register the new model if it's an improvement

## Notes

### Visualizations

Following include a list of visualization generated during validation, SHAP analysis and model sensitivity analysis which are present in dags/artifacts folder

1. learning_rate_sensitivity_xgboost.png
2. max_depth_sensitivity_xgboost.png
3. n_estimators_sensitivity_randomforest.png
4. n_estimators_sensitivity_xgboost.png
5. pm25_actual_vs_predicted_Prophet.png
6. pm25_actual_vs_predicted_RandomForest.png
7. pm25_actual_vs_predicted_XGBoost.png
8. shap_summary_plot_prophet.png
9. shap_summary_plot_randomforest.png
10. shap_summary_plot_xgboost.png
11. y_pred_prophet_day_of_week_MAE.png
12. y_pred_prophet_day_of_week_MBE.png
13. y_pred_prophet_day_of_week_RMSE.png
14. y_pred_prophet_day_of_week_R².png
15. y_pred_prophet_hour_MAE.png
16. y_pred_prophet_hour_MBE.png
17. y_pred_prophet_hour_RMSE.png
18. y_pred_prophet_hour_R².png
19. y_pred_prophet_month_MAE.png
20. y_pred_prophet_month_MBE.png
21. y_pred_prophet_month_RMSE.png
22. y_pred_prophet_month_R².png
23. y_pred_prophet_season_MAE.png
24. y_pred_prophet_season_MBE.png
25. y_pred_prophet_season_RMSE.png
26. y_pred_prophet_season_R².png
27. y_pred_rf_day_of_week_MAE.png
28. y_pred_rf_day_of_week_MBE.png
29. y_pred_rf_day_of_week_RMSE.png
30. y_pred_rf_day_of_week_R².png
31. y_pred_rf_hour_MAE.png
32. y_pred_rf_hour_MBE.png
33. y_pred_rf_hour_RMSE.png
34. y_pred_rf_hour_R².png
35. y_pred_rf_month_MAE.png
36. y_pred_rf_month_MBE.png
37. y_pred_rf_month_RMSE.png
38. y_pred_rf_month_R².png
39. y_pred_rf_season_MAE.png
40. y_pred_rf_season_MBE.png
41. y_pred_rf_season_RMSE.png
42. y_pred_rf_season_R².png
43. y_pred_xgb_day_of_week_MAE.png
44. y_pred_xgb_day_of_week_MBE.png
45. y_pred_xgb_day_of_week_RMSE.png
46. y_pred_xgb_day_of_week_R².png
47. y_pred_xgb_hour_MAE.png
48. y_pred_xgb_hour_MBE.png
49. y_pred_xgb_hour_RMSE.png
50. y_pred_xgb_hour_R².png
51. y_pred_xgb_month_MAE.png
52. y_pred_xgb_month_MBE.png
53. y_pred_xgb_month_RMSE.png
54. y_pred_xgb_month_R².png
55. y_pred_xgb_season_MAE.png
56. y_pred_xgb_season_MBE.png
57. y_pred_xgb_season_RMSE.png
58. y_pred_xgb_season_R².png

But these files may not be visible when ran through docker as they are executed in the docker environment, hence you can refer to the dags/artifacts folder for these images which were obtained by running the scripts using dags. In future this process will be migrated to Google Cloud.




## Model Deployment

Initially all the tasks in the Data Pipeline and the Model Development pipeline were run locally to get results. During the Deployment phase of the project we have moved these pipelines into the cloud infrastructure along with automating the training, retraining(in case of data drift/model decay) and deployment processes which are explained in the following sections.


### GitHub Actions automation

GitHub Actions is a powerful CI/CD (Continuous Integration/Continuous Deployment) tool that enables automation of workflows directly within your GitHub repository. It allows developers to define custom workflows using YAML files, making it flexible and highly configurable. These workflows can be triggered by various events, such as code pushes, pull requests, or even scheduled timings. The scheduled timings are done  using cron scheduling as per the requirements such as every 30 minutes, 1 hour or daily. 


## Limitation of GitHub workflow scheduling

Using the cron scheduling given by GitHub actions leads to limitation of number of concurrent jobs to just only 20 which because of restrictions from the free subscription by GitHub. 


To overcome this we use combination of Github Actions and Google Cloud Scheduler to automate the deployment process which will be covered in the future sections.


### Pipeline.yml

This GitHub actions file is configured to trigger the whole end to end pipeline when any changes are made in the main GitHub repository. It executes the following jobs in order.
There are 5 main stages in this GIthub Actions yaml file which are as follows:

**1. Job: Fetch-data**

o	The workflow uses environment variables like GCP_PROJECT_ID, IMAGE_NAME, and GCS_BUCKET to configure Google Cloud settings.

o	The repository is checked out using actions/checkout, ensuring access to all the code.

o	It installs the Google Cloud SDK, allowing interaction with Google Cloud resources like GCR and Cloud Functions.

o	Authenticates with Google Cloud using a service account key stored in GitHub Secrets (GCP_SA_KEY).

o	Installs Python dependencies from requirements.txt.

o	The fetch_data.py script is executed to collect the necessary data from the source.


**2. Job: Trigger-dag**

o	This job is triggered after the fetch-data job completes successfully (needs: fetch-data).

o	Installs curl and jq, utilities required to interact with APIs and parse JSON responses.

o	A unique DAG run ID is created using the current timestamp (datapipeline_YYYYMMDDHHMMSS), ensuring that each DAG run can be individually identified.

o	The workflow uses the Airflow REST API to trigger a DAG (datapipeline) by sending a POST request to Airflow’s server (AIRFLOW_VM_IP).

o	The unique DAG run ID generated earlier is passed in the request to start a specific DAG execution.

o	A polling mechanism checks the status of the DAG run by calling the Airflow API. It continuously checks the state (success, failed, or running).

o	The script waits until the DAG completes or fails and terminates the job accordingly.


**3. Job: Deploy-pipeline**

o	This job runs after the trigger-dag job finishes (needs: trigger-dag).

o	Repeats the setup and authentication process for accessing Google Cloud.

o	Configures Docker to authenticate with Google Container Registry (GCR) for pushing images.

o	Builds a Docker image for the application, specifying the base image and any required build arguments.

o	The Docker container runs a model training process by executing code inside the container(files in the cloud_run folder used for model development), with GCP credentials mounted for authentication.

o	Once the image is built and trained, it is pushed to the GCR with both the current IMAGE_TAG (using GitHub SHA) and latest tags.

o	Cleans up the temporary credentials file used for Google Cloud authentication.

o	After deployment completes (or fails), an email is sent with the status of the deployment pipeline. SMTP credentials are fetched from GitHub Secrets and used to send the email to the specified recipient.


**4. Job: Deploy-endpoint**

o	Runs after the deploy-pipeline job is complete (needs: deploy-pipeline).

o	Similar to the previous jobs, the repository is checked out.

o	The SDK is configured again to interact with Google Cloud.

o	Authenticates using the service account key, enabling access to GCP resources.

o	This step deploys the application as a Cloud Function (predict-function) using Google Cloud SDK. The function is triggered via HTTP requests, and the source code is deployed from the ./cloud_function directory.

o	After deploying the Cloud Function, an email is sent notifying the status of this deployment.


**5. Job: deploy-streamlit**

o	This job runs after deploy-endpoint completes (needs: deploy-endpoint).

o	The repository is checked out to access the Streamlit app’s code.

o	The Cloud SDK is installed and authenticated, as in previous jobs.

o	Ensures that Docker is authenticated with Google Container Registry for image pushes.

o	Builds the Docker image for the Streamlit application from the ./application directory, which will be used to deploy the app.

o	The built image is pushed to GCR with both version-specific and latest tags.

o	Deploys the Streamlit app to Google Cloud Run, ensuring that it is publicly accessible via HTTP requests on port 8080.

o	Finally, an email is sent to notify about the completion of the Streamlit deployment, detailing the success or failure.


**Overall Workflow Flow:**

•	**Data Fetching → Trigger Airflow DAG → Build & Deploy Pipeline → Deploy to Google Cloud Functions → Deploy Streamlit App to Cloud Run**

•	At each stage, email notifications are sent to the specified recipient, updating them about the status of the job. These notifications are sent regardless of the job outcome, thanks to the if: always() condition in each email step.

•	Email notifications are sent after every  success/failure of each job, in case of failure the subsequent jobs are halted.


 Now we shall go through all the scripts triggered during the execution of this yaml file

## Fetch_data.py

 o	Configures API authentication and specifies the target city, country, and time range for data collection.
 
o	Fetches the location ID corresponding to the city and retrieves air pollution data for the specified time range.

o	Collects available air quality parameters (e.g., PM2.5, CO) for additional insights.

o	Processes the raw data to create a structured format for analysis and storage.

o	Identifies and logs anomalies in the data, such as missing fields, invalid values (e.g., negative pollution levels), and duplicate records.

o	Checks if the processed data already exists in the GCS bucket to avoid redundancy.

o	Saves the validated data as a CSV file directly to GCS for further use.



**Cloud_run folder:**

Here a detailed description of all files under the Cloud_run folder which are used for model development is given

## Docker_file:

This Dockerfile is designed for a streamlined, efficient Python environment to handle the various machine learning tasks involved in the model development process. Here's an overview of its key features and workflow:

1. Base Image: It uses a minimal Python environment (python:3.8-slim) to keep the container lightweight.

2. Environment Configuration:

    1. Sets up GOOGLE_APPLICATION_CREDENTIALS for authenticating with Google Cloud services.
        
    2. Configures MLFLOW_TRACKING_DIR for managing MLflow experiment logs and artifacts.
       
3. System Dependencies: Installs necessary system-level packages like gcc, libc-dev, and jq for compatibility and scripting needs.
4. Python Dependencies: Installs all required Python libraries listed in requirements.txt.
5. Code Setup:	

    1. Copies multiple Python scripts into the container, each responsible for specific tasks such as data processing, model training, validation, and evaluation.
 
    2. Grants full permissions to the MLflow directory to ensure seamless logging and artifact storage.

6. Workflow Execution: The container is configured to sequentially execute a series of Python scripts that collectively handle the following tasks:
   
**Data Management:**
1. delete_table.py: Deletes existing data in Google BigQuery.
2. saving_bigquery.py: Saves new data to BigQuery.
    
**Model Training and Validation:**
1. Prophet_train.py and Prophet_Valid.py: Train and validate the Prophet model.
2. XGBoost_train.py and XGBoost_valid.py: Train and validate the XGBoost model.
3. random_forest_train.py and RandomForest_Valid.py: Train and validate the Random Forest model.
    
**Model Evaluation:**
1. Model_bias.py: Evaluates model bias to ensure fairness and reliability.
2. bestmodel.py: Determines and finalizes the best-performing model.



## delete_table.py

•	Deletes the specified BigQuery table (airquality-438719.airqualityuser.allfeatures).

•	Introduces a delay to ensure the deletion is propagated.

•	Verifies whether the table still exists and confirms deletion.

•	Provides error handling for issues encountered during the process.


## saving_bigquery.py

This script is designed to load, process, and insert feature-engineered data (in pickle format) from GCS into BigQuery, where it can be used for further analysis or model training. It ensures proper handling of timestamps and performs batch insertions to optimize data loading. Additionally, it checks and creates the target BigQuery table if necessary.

•	Creates a BigQuery client to interact with the BigQuery project (airquality-438719).

•	Initializes a Cloud Storage client to access data from a GCS bucket (airquality-mlops-rg).

•	Specifies paths to feature-engineered data stored in GCS (test and train sets in pickle format).

•	Downloads the pickle files containing the feature-engineered data from the specified GCS bucket.

•	Deserializes the files using pickle to load the data into Python 

•	Ensures that the timestamp column is correctly formatted as datetime and is present in the dataset.

•	Converts the timestamp to ISO 8601 format and prepares the feature data for BigQuery

•	Converts each row of data into a dictionary containing various features like PM2.5, rolling means, and other time-based features.

•	Each dictionary is then serialized into a JSON string and appended to a list

•	The data is uploaded in batches of 1000 rows to BigQuery using the insert_rows_json method.

•	The script handles errors during insertion and prints success or error messages accordingly.

•	Before inserting data, the script checks if the target table (airquality-438719.airqualityuser.allfeatures) exists in BigQuery.

•	If the table doesn’t exist, it creates the table with the defined schema:

1. timestamp (nullable TIMESTAMP).
2. feature_data (nullable STRING for storing JSON data)
    
•	Finally, the script calls the function to insert data for both test and train datasets into the BigQuery table


## Prophet_train.py


The script automates the process of training a PM2.5 prediction model using Prophet, while logging key metrics and the trained model itself in MLflow, and ensuring that the model is saved for future use in Google Cloud Storage. This approach supports model versioning, monitoring, and deployment in a cloud-based environment

•	Loads training data from a pickle file stored in a GCS bucket, deserializing it into a Pandas DataFrame.

•	Prepares the training data in the format expected by Prophet (a DataFrame with ds for timestamps and y for PM2.5 values).

•	Trains the Prophet model using the preprocessed data and logs the model to MLflow.

•	The trained model's weights are serialized with pickle and uploaded to GCS for storage.


## Prophet_Valid.py


•	Initializes MLflow for experiment tracking and logs parameters and metrics.

•	Loads training and test data from Google Cloud Storage.

•	Applies feature engineering, including transformations and generating time-series features (lags, rolling means, etc.).

•	Handles skewness in the target variable (PM2.5) using Box-Cox or log transformations.

•	Downloads the pre-trained Prophet model from GCS and loads it using pandas.read_pickle

•	 Makes predictions on the test data and calculates performance metrics like RMSE (Root Mean Squared Error).

•	Performs SHAP (Shapley values) analysis to interpret the model's feature importance and saves the SHAP summary plot.

•	Plots actual vs. predicted PM2.5 values and saves the plot as an artifact in MLflow.

•	Saves and uploads artifacts such as SHAP plots and result plots to Google Cloud Storage for later use.


## XGBoost_train.py


•	The script imports various libraries like XGBoost, MLflow, Google Cloud Storage, and others for machine learning, data processing, and experiment tracking

•	The setup_mlflow_tracking() function configures the MLflow tracking URI, where experiment logs and metrics will be stored locally.

•	Defines model attributes, including hyperparameter grid and placeholders for training data

•	Downloads training data from a Google Cloud Storage bucket, processes it, and splits it into 
features (X_train) and target (y_train).

•	Performs a grid search for optimal hyperparameters for the XGBoost model, using cross-validation and logging the results in MLflow.

•	Fits the XGBoost model on the training data and logs the trained model in MLflow.

•	Saves the trained model's weights to a specified Google Cloud Storage bucket.


## XGBoost_valid.py

•	Loads PM2.5 data from Google Cloud Storage, performs skewness checks, and applies feature engineering. This includes creating lag features, rolling statistics, and cyclical features (like hour of day or day of the week). The target variable (pm25) is also transformed if necessary using a Box-Cox transformation.

•	Sets up an XGBoost regression model for predicting PM2.5. The model is trained using features engineered in the previous step. Hyperparameter tuning is performed using grid search for parameters like n_estimators, learning_rate, and max_depth. The training and evaluation are tracked using MLflow.

•	The script evaluates how changes in hyperparameters (e.g., n_estimators, learning_rate, max_depth) affect model performance (measured by RMSE). The results are logged to MLflow and visualized with a plot

•	SHAP (Shapley Additive Explanations) is used to interpret the model's predictions, providing insights into the importance of different features.

•	The model’s performance is evaluated on a test set, and RMSE is calculated and logged. The actual vs. predicted PM2.5 values are plotted and saved as artifacts in MLflow and Google Cloud Storage.


## Random_forest_train.py

•	The script uses MLflow to track experiments, log parameters, and save model artifacts (such as hyperparameters, metrics, and the trained model itself).

•	It sets up the MLflow tracking URI to store logs locally, ensuring all training details are stored in the specified directory.

•	The data is fetched from Google Cloud Storage (GCS), where a pickled dataset containing feature-engineered data (feature_eng_data.pkl) is downloaded.

•	It uses pickle to deserialize the data into a pandas DataFrame and extracts the target variable (pm25) and features for training the model.

•	The model is a Random Forest Regressor from sklearn, which is initialized with specific hyperparameters (e.g., random_state=42 for reproducibility).

•	A grid search with cross-validation (GridSearchCV) is performed to tune the model's hyperparameters, specifically n_estimators, and the best parameters are selected based on mean squared error

•	The model is trained on the feature data (X_train) and the target variable (y_train), and the training process duration is logged as an MLflow metric.

•	After training, the best model is serialized using pickle and uploaded to Google Cloud Storage for persistence. The model weights are saved under the weights/rf_model.pth path in the cloud.

•	The model is also logged as an artifact in MLflow, making it easy to track and load in future runs.

•	Hyperparameters (from the grid search) are logged using mlflow.log_params(), and the model is logged as a MLflow artifact. The training duration is tracked as a metric (training_duration).


## Random_forest_valid.py

•	Configures MLflow for experiment tracking, saving logs locally with a specified tracking URI.

•	Handles data loading, feature engineering, and skewness checks, creating lag features, rolling statistics, and time-based features

•	Manages training, evaluation, hyperparameter tuning, SHAP analysis, and result plotting for the Random Forest model.

•	Logs and evaluates the impact of hyperparameters (e.g., n_estimators) on model performance using RMSE.

•	Uses SHAP for model interpretability, visualizing feature importance and contributions to predictions.

•	Computes RMSE for model performance, with optional inverse transformations for skewed data.

•	Plots actual vs predicted PM2.5 values and logs them as artifacts in both MLflow and Google Cloud Storage.


## Model_bias.py

•	The script loads feature-engineered test data from Google Cloud Storage, processing it by adding missing columns (season, timestamp) and dropping irrelevant ones (pm25_boxcox, pm25_log).

•	The script defines functions to load pre-trained models (Random Forest, XGBoost, Prophet) from Google Cloud Storage, ensuring each model is available for predictions.

•	The models (Random Forest, XGBoost, Prophet) are used to make predictions on the feature data, which are stored as new columns in the dataset (y_pred_rf, y_pred_xgb, y_pred_prophet).

•	The evaluate_model_bias() function evaluates model bias by analyzing performance across different slicing features like hour, day_of_week, month, and season.

•	For each slice of the slicing features, the script calculates MAE, RMSE, R², and MBE to assess model performance in each segment.

•	Metrics for each model and slicing feature are logged to MLflow, ensuring easy tracking of performance and bias across experiments.

•	The function computes deviations from average metrics for each slice and flags slices as biased if the deviation exceeds a threshold.

•	The biased metrics (e.g., biased MAE, RMSE) are logged in MLflow to track the count of biased slices for each model and slicing feature.

•	The distribution of bias metrics is visualized using bar plots, which are saved locally and uploaded to Google Cloud Storage for easy access and review.


## Best_model.py

•	The script evaluates multiple machine learning models (Random Forest, XGBoost, Prophet) across different experiments based on RMSE and bias metrics to select the best model.

•	A combined score for each model is computed as a weighted sum of RMSE and bias metrics, where RMSE is given more importance (0.5 weight), and bias metrics like MAE, R2, and MBE are assigned lower weights.

•	Weights for the bias metrics are defined (MAE: 0.2, R2: 0.2, MBE: 0.1), reflecting their importance in the combined score calculation.

•	The bias results (e.g., MAE, RMSE, R2, MBE) are collected for different features (hour, day_of_week, etc.) to assess how each model performs across these features.

•	 The best model is selected based on the combined score, which factors in both RMSE and bias metrics. The model with the lowest combined score is chosen.

•	 Before uploading a new model, the script checks the existing RMSE value from the registry. If the new model’s RMSE is worse, it does not upload the new model, maintaining the best-performing one.

•	The script loads pre-trained models (Random Forest, XGBoost, Prophet) from Google Cloud Storage (GCS) by downloading model weights to temporary files and then deserializing them.

•	After selecting the best model, the script uploads the model weights to GCS, along with the updated RMSE value, ensuring only the best model is saved.

•	There are mechanisms in place to catch errors during model loading (e.g., incorrect model format) and ensure models are loaded correctly before further processing.

•	The script uses MLflow’s model registry to check for existing compares them before deciding to overwrite or update the stored model.


## Data Drift and Model Decay

This section explains key concepts like model decay and data drift, followed by how they are addressed in the project. It integrates actual implementation details, including Google Cloud components, monitoring, automation, and retraining workflows.

**What is Model Decay?**

Definition:Model decay occurs when a machine learning model's performance deteriorates over time due to changes in real-world conditions that differ from the training environment. Common reasons include:

- Data Drift: The input data distribution changes over time.
- Concept Drift: The relationship between input features and output labels changes.

**Why it Matters**

Model decay leads to inaccurate predictions, impacting business decisions and operational efficiency. Regular monitoring is crucial to detect and address decay early.
What We Are Doing in This Project

1.	Monitoring Key Metrics:
    - Metrics such as RMSE (Root Mean Square Error) are monitored in real-time by comparing predictions with observed values.
    - Thresholds Set: 
        - Prediction Difference > 3: Retraining is triggered if the absolute difference between the predicted and actual values exceeds this value.      
2.	Real-Time Monitoring Infrastructure:
    - Google Monitoring: 
        - Metrics such as prediction error (difference between predicted and actual values) are logged using Google Monitoring.
        - Alerts are configured in Google Monitoring to notify the system when thresholds are breached.
3. BigQuery: 
    - Predictions and input features are stored in BigQuery for further analysis and decay detection.
4. GitHub Actions: 
    - Automates the retraining pipeline when decay is detected.
5. Decay Detection:
    - Using compare_and_trigger_with_temp_table, the project identifies whether differences between predictions and observed values exceed the set threshold.
6. Automated Functions:
    - A Google Cloud Function (trigger_github_pipeline_model) is triggered when decay is detected, initiating the retraining workflow through GitHub Actions.
    - This Cloud Function interacts with BigQuery to fetch predictions and feature data, compare them, and determine if retraining is necessary.
7. Cloud Scheduler:
    - A Cloud Scheduler job runs periodically (e.g., hourly or daily) to trigger the decay detection workflow, ensuring timely identification of issues.


**What is Data Drift?**
   
Definition:Data drift occurs when the statistical properties of input data change over time compared to the training data. This can reduce the model's predictive performance.

**Why it Matters**

Even if the model initially performs well, data drift can affect its ability to generalize to new data. Detecting and addressing drift ensures the model remains accurate and reliable.

**What We Are Doing in This Project**

1.	Detecting Data Drift:
    - Drift is detected by comparing the mean and variance of new data with the existing data using the detect_drift function: 
    - drift_value = mean_diff + var_diff
    - Drift detection checks are implemented via an Airflow DAG.
2.	Thresholds Set:
    - Drift Value > 3: If the combined mean and variance differences exceed 3, the retraining workflow is triggered.
3.	Recording and Visualizing Drift:
    - Recording: 
            - Drift values are logged in a GCS file (api_data/drift.txt) for historical tracking.
    - Plotting: 
            - A visualization of drift over time is generated using Matplotlib and saved to GCS as a plot (api_data/plot_drift.png).
4.	Automated Alerts:
    - GitHub Actions monitors drift metrics and triggers retraining workflows when thresholds are exceeded.
    - Notifications are sent to stakeholders to inform them of detected drift.
5.	Automated Functions:
    - A Google Cloud Function (trigger_github_pipeline) is triggered when data drift is detected.
    - This function calculates drift metrics, logs the drift values, and determines whether to trigger the GitHub Actions pipeline for retraining.
6.	Cloud Scheduler:
    - A Cloud Scheduler job periodically triggers the drift detection workflow to ensure timely monitoring.
  	

**What Are Thresholds for Triggering Retraining?**

Definition:Thresholds are predefined limits for performance metrics and data drift values. If these thresholds are breached, the retraining process is triggered.

**Why it Matters**

Thresholds ensure the model is retrained only when necessary, avoiding unnecessary computational costs while maintaining optimal performance.

**What We Are Doing in This Project**

1.	Performance Metrics Thresholds:
    - Prediction Difference > 3: Retraining is triggered if the absolute difference between the predicted and actual values exceeds this value.
2.	Drift Metrics Thresholds:
    - Drift Value > 3: If the combined mean and variance differences exceed 3, retraining is triggered.
3.	Integration:
    - Threshold checks are automated using Evidently AI, GitHub Actions, and Airflow.
    - If a threshold is breached, an Airflow DAG is triggered to initiate retraining.

**Automating the Retraining Pipeline**

Definition:An automated retraining pipeline continuously evaluates model performance and initiates steps like data pulling, retraining, validation, and redeployment when thresholds are breached.

**Why it Matters**

Automation minimizes manual effort, ensures timely retraining, and reduces downtime caused by decayed models.

**What We Are Doing in This Project**

1.	Pulling New Data:
    - Latest data is fetched automatically from GCS using the fetch_latest_csv function.
    - Data is preprocessed to ensure compatibility with the training pipeline.
2.	Retraining the Model:
    - A Dockerized training pipeline retrains the model using the latest data.
    - Retraining is performed within a secure and consistent environment.
3.	Validating the Model:
    - New model performance is evaluated against existing metrics such as accuracy and RMSE.
    - The new model is deployed only if it outperforms the current model.
4.	Deploying the Model:
    - Deployment to production is automated using GCP Cloud Functions.
    - The current model is retained if the new model fails validation.
  	

**Notifications for Model Retraining**

Definition:Notifications alert stakeholders about retraining triggers, completion, or any anomalies detected during the pipeline.

**Why it Matters**

Keeping stakeholders informed ensures transparency, aids compliance, and allows for quick resolution of any issues that may arise post-deployment.

**What We Are Doing in This Project**

1.	Email Notifications:
    - Notifications are sent via email when
        - Retraining is triggered.
        - Retraining is completed (success or failure).
        -  A new model is deployed or the existing one is retained.
    - The SMTP email service is configured in GitHub Actions for notification delivery.
2.	Details Shared in Notifications:
    - Metrics before and after retraining.
    - Reasons for retraining (e.g., data drift or model decay).
    - Deployment status.
3.	Future Enhancements:
    - Integration with Slack or Microsoft Teams for real-time notifications and collaboration.















