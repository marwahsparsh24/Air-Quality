import pandas as pd
import os
import logging

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.pivoted_data = None

    def load_data(self):
        # Check if the file exists
        if not os.path.exists(self.file_path):
            logger.error(f"File '{self.file_path}' does not exist.")
            return ["File does not exist"]
        
        # Load the data
        self.data = pd.read_pickle(self.file_path)
        logger.info(f"Data loaded from {self.file_path}")
        return []

    def check_column_consistency(self):
        # Check for the presence of required columns
        required_columns = {'date', 'parameter', 'value'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            anomaly = f"Missing required columns: {', '.join(missing_columns)}."
            logger.error(anomaly)
            return [anomaly]
        
        logger.info("All necessary columns are present.")
        return []

    def detect_date_anomalies(self):
        anomalies = []
        
        # Check if 'date' column exists
        if 'date' not in self.data.columns:
            anomaly = "Missing 'date' column in DataFrame."
            logger.error(anomaly)
            anomalies.append(anomaly)
            return anomalies
        
        # Convert 'date' column to datetime and handle timezone compatibility
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        
        # Make sure current timestamp has the same timezone as 'date' column
        current_timestamp = pd.Timestamp.now(tz=self.data['date'].dt.tz)
        
        # Check for future dates
        future_dates = self.data[self.data['date'] > current_timestamp]
        
        if not future_dates.empty:
            anomaly = "Future dates detected in 'date' column."
            logger.warning(anomaly)
            anomalies.append(anomaly)
        
        logger.info("Date column processed successfully.")
        return anomalies

    def pivot_data(self):
        if 'date' not in self.data.columns or 'parameter' not in self.data.columns or 'value' not in self.data.columns:
            raise ValueError("Missing one or more required columns: 'date', 'parameter', 'value'.")
        
        self.pivoted_data = self.data.pivot_table(index='date', columns='parameter', values='value').reset_index()
        logger.info("Data pivoted successfully.")
        return []

    # def pivot_data(self):
    #     # Pivot data
    #     self.pivoted_data = self.data.pivot_table(index='date', columns='parameter', values='value').reset_index()
    #     logger.info("Data pivoted successfully.")
    #     return []

    def save_as_pickle(self, output_path):
        if self.pivoted_data is not None:
            self.pivoted_data.to_pickle(output_path)
            logger.info(f"Pivoted DataFrame saved as '{output_path}'.")
            return []
        else:
            logger.error("No pivoted data to save. Please pivot the data first.")
            return ["No pivoted data to save"]
    
    def process_dates(self):
        if 'date' not in self.data.columns:
            raise ValueError("No 'date' column in the DataFrame.")
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        print("Date column converted to datetime.")

def pivot_parameters():
    file_path = os.path.join(os.getcwd(), "DataPreprocessing/src/data_store_pkl_files/train_data/train_data.pkl")
    output_pickle_file = os.path.join(os.getcwd(), "DataPreprocessing/src/data_store_pkl_files/train_data/pivoted_train_data.pkl")
    processor = DataProcessor(file_path)
    
    anomalies = []
    # Step 1: Load data and check for file existence anomalies
    anomalies.extend(processor.load_data())
    
    # Step 2: Check for column consistency
    anomalies.extend(processor.check_column_consistency())

    # Step 3: Detect anomalies in the 'date' column
    anomalies.extend(processor.detect_date_anomalies())

    # Step 4: Pivot data if no critical anomalies are detected
    if not anomalies:
        anomalies.extend(processor.pivot_data())
        anomalies.extend(processor.save_as_pickle(output_pickle_file))
        logger.info("Data processing and pivoting completed successfully.")
    else:
        logger.error("Anomalies detected; skipping pivoting and saving.")
    
    return anomalies  # Return list of all detected anomalies

if __name__ == "__main__":
    detected_anomalies = pivot_parameters()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
