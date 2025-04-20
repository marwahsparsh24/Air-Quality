import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
import logging
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(f"Data loaded from {self.file_path}.")
    
    def handle_missing_values(self):
        anomalies = []
        
        # Check for any missing values in the dataset
        missing_summary = self.data.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            logger.warning(f"Missing values found: {missing_summary[missing_summary > 0].to_dict()}")
            anomalies.append(f"Total missing values: {total_missing}")
            
            if 'pm25' in self.data.columns and self.data['pm25'].isnull().any():
                self.data['pm25'] = self.data['pm25'].interpolate(method='linear')  # Apply linear interpolation
                logger.info("'pm25' missing values interpolated.")
            else:
                logger.warning("'pm25' column not found for interpolation or no missing values in 'pm25'.")
        else:
            logger.info("No missing values found.")
        
        return anomalies

    def save_as_pickle(self, output_path):
        anomalies = []
        
        # Check if data is available to save
        if self.data is None:
            anomaly = "No data available to save. Please load and process the data first."
            logger.error(anomaly)
            anomalies.append(anomaly)
            return anomalies

        # Check if the output path directory exists, create if not
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"Created directory for saving file: {output_dir}")
            except Exception as e:
                anomaly = f"Failed to create directory {output_dir}: {e}"
                logger.error(anomaly)
                anomalies.append(anomaly)
                return anomalies

        # Attempt to save the data to pickle
        try:
            self.data.to_pickle(output_path)
            logger.info(f"Processed DataFrame saved as '{output_path}'.")
        except Exception as e:
            anomaly = f"Failed to save data to '{output_path}': {e}"
            logger.error(anomaly)
            anomalies.append(anomaly)

        return anomalies

def handle_missing_vals():
    # Path to the input pickle file and output pickle file
    file_path = os.path.join(os.getcwd(),"dags/DataPreprocessing/src/data_store_pkl_files/train_data/cleaned_train_data.pkl")
    output_pickle_file = os.path.join(os.getcwd(),"dags/DataPreprocessing/src/data_store_pkl_files/train_data/no_null_train_data.pkl")
    processor = DataProcessor(file_path)
    processor.load_data()
    anomalies = []
    anomalies.extend(processor.handle_missing_values())
    anomalies.extend(processor.save_as_pickle(output_pickle_file))
    return anomalies

if __name__ == "__main__":
    detected_anomalies = handle_missing_vals()
    if detected_anomalies:
            logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
