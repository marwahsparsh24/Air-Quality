# Air-Quality
 This repository contains the code and resources for predicting PM2.5 and PM10 levels using air quality data from OpenAQ. It includes data collection, preprocessing, and modeling scripts along with analysis and visualizations. The project aims to forecast PM2.5 and PM10 levels and provide insights into air pollution trends.

## pre-requisites

git

python>=3.8

docker desktop is running

        
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
```bash
docker-compose up airflow-init
```
#### Step 5: Start the Airflow Services
```bash
docker-compose up -d
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

#### Step 8: Stopping the Airflow Services
``` bash
docker-compose down
```
       
### Folder Structure Overview

```plaintext
Air-Quality/
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
├── ModelDeployment/
├── ModelDevelopment/
├── ModelLogging/
├── ModelMonitoring/
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

README.md:                        Project documentation file.


## Data Pipeline Steps

The data pipeline for the Air-Quality project consists of a series of steps to preprocess, transform, and model air quality data. Each stage of the pipeline performs a specific function that prepares the data for PM2.5 and PM10 predictions.
 
 ### Pipeline Steps

#### 1. Data Collection

This step collects raw air quality data from OpenAQ API and stores it locally. The script is designed to fetch historical PM2.5 and PM10 readings based on specified parameters, such as geographic region and time range. Also stacks csv files

 Scripts: 
 
  dags/DataPreprocessing/src/data_loader.py
  dags/DataPreprocessing/src/data_air.py


#### 2.Schema Validation

Ensures that the collected data adheres to a predefined schema. This validation step is crucial to check for consistency and correctness in the dataset structure. The schema includes necessary columns and data types for downstream processing.

Script: 

dags/DataPreprocessing/src/Schema/check_schema_original_airpollution.py


#### 3.Data Preprocessing

In this step, the data is cleaned and prepared for analysis. It includes handling missing values, detecting anomalies, and performing initial feature engineering. The preprocessed data is stored as .pkl files in data_store_pkl_files for both training and testing sets.

Scripts:

 dags/DataPreprocessing/src/data_split.py
 dags/DataPreprocessing/train/data_preprocessing/pivoting_data.py
 dags/DataPreprocessing/train/data_preprocessing/removal_of_uneccesary_cols.py
 dags/DataPreprocessing/train/data_preprocessing/check_missing_values.py
 dags/DataPreprocessing/train/data_preprocessing/anamoly_detection.py
            

#### 4.Feature Engineering

This feature engineering step enhances model performance by capturing temporal patterns and environmental influences essential for air quality prediction. Techniques applied include lag and lead features to account for past and immediate future pollutant levels, rolling statistics (mean, sum, min, max) to summarize recent trends, and differencing to highlight rate of change. Cosine similarity with historical patterns is used to identify recurring pollution trends, while time-based features (hour, day, month) help capture cyclical variations providing a robust set of features for effective PM2.5 and PM10 predictions.

Script:

dags/DataPreprocessing/train/data_preprocessing/feature_eng.py

#### 5. Data Validation

Runs validation checks to ensure the test,train data meets quality standards before entering the modeling phase. This includes confirming the absence of missing values and verifying data transformations.

Script: 
      
dags/DataPreprocessing/test/data_preprocessing/check_output_data_schema.py

#### 6. Data Bias

Analyzes potential biases in the data that could impact model predictions. It is checked multiple times through out the pipeline

Script:
 
dags/DataPreprocessing/src/data_bias_check_final.py
