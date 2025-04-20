import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, pickle_file_path):
        self.pickle_file_path = pickle_file_path
        self.dataframe = None
        self.train_df = None
        self.test_df = None

    def load_pickle(self):
        if not os.path.exists(self.pickle_file_path):
            logger.error(f"Pickle file '{self.pickle_file_path}' does not exist.")
            return ["Pickle file not found"]
        
        self.dataframe = pd.read_pickle(self.pickle_file_path)
        logger.info(f"Loaded data from {self.pickle_file_path}.")
        if self.dataframe.empty:
            logger.warning("Loaded DataFrame is empty.")
            return ["Loaded DataFrame is empty"]
        return []

    def detect_anomalies(self):
        anomalies = []
        if self.dataframe is None:
            anomalies.append("No DataFrame loaded.")
            logger.error("No DataFrame loaded.")
            return anomalies
        
        # Check for essential columns (if known in advance)
        required_columns = ["date", "parameter", "value"]
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            anomaly = f"Missing columns in DataFrame: {', '.join(missing_columns)}"
            logger.error(anomaly)
            anomalies.append(anomaly)
        
        # Check for empty DataFrame after loading
        if self.dataframe.empty:
            anomalies.append("DataFrame is empty.")
            logger.error("DataFrame is empty.")
        
        return anomalies

    def split_data(self, test_size=0.2, random_state=42):
        if self.dataframe is None:
            logger.error("No DataFrame to split. Please load the pickle file first.")
            return ["No DataFrame to split"]

        self.train_df, self.test_df = train_test_split(self.dataframe, test_size=test_size, random_state=random_state)
        if self.train_df.empty or self.test_df.empty:
            logger.warning("Split resulted in an empty train or test set.")
            return ["Empty train or test set after split"]
        
        logger.info(f"Data split into training (size={len(self.train_df)}) and testing (size={len(self.test_df)}) sets.")
        return []

    def save_as_pickle(self, train_output_path, test_output_path):
        if self.train_df is None or self.test_df is None:
            logger.error("No split data to save. Please split the data first.")
            return ["No split data to save"]
        
        # Verify paths
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        
        self.train_df.to_pickle(train_output_path)
        self.test_df.to_pickle(test_output_path)
        logger.info(f"Training DataFrame saved as '{train_output_path}' and Testing DataFrame saved as '{test_output_path}'.")
        return []

def split():
    pickle_file_path = os.path.join(os.getcwd(), "dags/DataPreprocessing/src/data_store_pkl_files/resampled_data.pkl")
    train_output_pickle_file = os.path.join(os.getcwd(), "dags/DataPreprocessing/src/data_store_pkl_files/train_data/train_data.pkl")
    test_output_pickle_file = os.path.join(os.getcwd(), "dags/DataPreprocessing/src/data_store_pkl_files/test_data/test_data.pkl")

    data_splitter = DataSplitter(pickle_file_path)
    anomalies = []

    # Step 1: Load and check for anomalies
    anomalies.extend(data_splitter.load_pickle())
    anomalies.extend(data_splitter.detect_anomalies())

    # Step 2: Split data and check for anomalies
    if not anomalies:
        anomalies.extend(data_splitter.split_data(test_size=0.2, random_state=42))
    
    # Step 3: Save split data and check for anomalies
    if not anomalies:
        anomalies.extend(data_splitter.save_as_pickle(train_output_pickle_file, test_output_pickle_file))

    if anomalies:
        logger.error(f"Anomalies detected during splitting: {anomalies}")
    else:
        logger.info("Data splitting completed successfully with no anomalies.")
    
    return anomalies

if __name__ == "__main__":
    detected_anomalies = split()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
