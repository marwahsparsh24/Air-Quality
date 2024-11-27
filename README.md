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
