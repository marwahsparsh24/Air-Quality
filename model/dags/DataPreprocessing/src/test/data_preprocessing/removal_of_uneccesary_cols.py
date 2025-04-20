import pandas as pd
import os
import logging

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            logger.error(f"File '{self.file_path}' does not exist.")
            return ["File does not exist"]

        self.data = pd.read_pickle(self.file_path)
        logger.info(f"Data loaded from {self.file_path}.")
        return []

    def drop_columns(self, columns_to_drop):
            if self.data is None:
                logger.error("Data is not loaded. Cannot drop columns.")
                return ["Data not loaded"]
            
            if self.data is not None:
                columns_present = [col for col in columns_to_drop if col in self.data.columns]
            
                if columns_present:
                    self.data.drop(columns=columns_present, inplace=True)
                    logger.info(f"Successfully dropped columns: {columns_present}")
                else:
                    logger.warning(f"None of the specified columns {columns_to_drop} exist in the DataFrame.")
                    return [f"None of the columns {columns_to_drop} exist in the DataFrame"]
                self.data.set_index('date',inplace=True)
            if self.data.empty:
                anomaly = "DataFrame is empty after dropping columns."
                logger.warning(anomaly)
                return [anomaly]
            return []
    
    def set_date_index(self):
        if 'date' not in self.data.columns:
            anomaly = "Missing 'date' column to set as index."
            logger.error(anomaly)
            return [anomaly]

        # Ensure 'date' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            try:
                self.data['date'] = pd.to_datetime(self.data['date'])
                logger.info("Converted 'date' column to datetime.")
            except Exception as e:
                anomaly = f"Failed to convert 'date' column to datetime: {e}"
                logger.error(anomaly)
                return [anomaly]

        # Set 'date' as index
        self.data.set_index('date', inplace=True)
        logger.info("Set 'date' column as index successfully.")
        return []

    def save_as_pickle(self, output_path):
        if self.data is None:
            anomaly = "No data available to save. Please load and clean the data first."
            logger.error(anomaly)
            return [anomaly]
        
        self.data.to_pickle(output_path)
        logger.info(f"Cleaned DataFrame saved as '{output_path}'.")
        return []

def remove_uneccesary_cols():
    # Paths for input and output pickle files
    file_path = os.path.join(os.getcwd(), "dags/DataPreprocessing/src/data_store_pkl_files/test_data/pivoted_test_data.pkl")
    output_pickle_file = os.path.join(os.getcwd(), "dags/DataPreprocessing/src/data_store_pkl_files/test_data/cleaned_test_data.pkl")
    columns_to_drop = ['co', 'no', 'no2', 'o3', 'so2']
    cleaner = DataCleaner(file_path)
    
    anomalies = []
    # Step 1: Load data
    anomalies.extend(cleaner.load_data())

    # Step 2: Drop unnecessary columns
    anomalies.extend(cleaner.drop_columns(columns_to_drop))

    # Step 3: Set 'date' column as index
    anomalies.extend(cleaner.set_date_index())

    # Step 4: Save the cleaned DataFrame
    if not anomalies:
        anomalies.extend(cleaner.save_as_pickle(output_pickle_file))
        logger.info("Data cleaning process completed successfully.")
    else:
        logger.error("Anomalies detected; skipping saving the cleaned data.")
    
    return anomalies  # Return a list of all detected anomalies

if __name__ == "__main__":
    detected_anomalies = remove_uneccesary_cols()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
