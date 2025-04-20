import pandas as pd
import os
import logging

# Configure logging for anomaly detection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVStacker:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.dataframes = []
        self.stacked_df = None

    def load_csv_files(self):
        if not os.path.exists(self.folder_path):
            logger.error(f"Folder '{self.folder_path}' does not exist.")
            return ["Folder does not exist"]

        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        if not csv_files:
            logger.warning(f"No CSV files found in '{self.folder_path}'.")
            return ["No CSV files found"]

        anomalies = []
        for file in csv_files:
            file_path = os.path.join(self.folder_path, file)
            if os.path.getsize(file_path) == 0:
                anomaly = f"File '{file}' is empty and was skipped."
                logger.warning(anomaly)
                anomalies.append(anomaly)
                continue

            df = pd.read_csv(file_path)
            self.dataframes.append(df)
            logger.info(f"Loaded file '{file}' successfully.")
        if not self.dataframes:
            anomalies.append("No valid data files were loaded.")
            logger.error("No valid data files were loaded.")
        return anomalies

    def detect_column_consistency(self):
        if not self.dataframes:
            logger.error("No DataFrames loaded to check for column consistency.")
            return ["No DataFrames loaded to check for column consistency"]

        anomalies = []
        first_columns = self.dataframes[0].columns
        for i, df in enumerate(self.dataframes[1:], start=1):
            if not df.columns.equals(first_columns):
                anomaly = f"Column mismatch detected in DataFrame {i}."
                logger.error(anomaly)
                anomalies.append(anomaly)
        if not anomalies:
            logger.info("All columns are consistent.")
        return anomalies

    def detect_data_anomalies(self):
        anomalies = []
        for i, df in enumerate(self.dataframes):
            for column in ["date", "parameter", "value"]:
                if column not in df.columns:
                    anomaly = f"Missing critical column '{column}' in DataFrame {i}."
                    logger.error(anomaly)
                    anomalies.append(anomaly)

            for _, row in df.iterrows():
                if row["parameter"] in ["pm25", "pm10", "o3", "no2", "so2", "co"] and row["value"] < 0:
                    anomaly = f"Invalid negative value detected for {row['parameter']}: {row['value']} on {row['date']} in DataFrame {i}."
                    logger.warning(anomaly)
                    anomalies.append(anomaly)
        return anomalies

    def stack_dataframes(self):
        if not self.dataframes:
            logger.error("No DataFrames to stack. Please load CSV files first.")
            return ["No DataFrames to stack"]
        
        self.stacked_df = pd.concat(self.dataframes, ignore_index=True)
        logger.info("DataFrames stacked successfully.")
        return []

    def save_as_pickle(self, output_path):
        if self.stacked_df is None:
            anomaly = "No DataFrame to save. Please stack the dataframes first."
            logger.error(anomaly)
            return [anomaly]
        
        self.stacked_df.to_pickle(output_path)
        logger.info(f"Stacked DataFrame saved as '{output_path}'.")
        return []

def stack_csvs_to_pickle():
    folder_path = "dags/DataPreprocessing/src/data_store_pkl_files/csv"
    output_pickle_file = "dags/DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl"

    csv_stacker = CSVStacker(folder_path)
    anomalies = []

    # Step 1: Load files and detect file-level anomalies
    anomalies.extend(csv_stacker.load_csv_files())

    # Step 2: Detect column consistency anomalies
    anomalies.extend(csv_stacker.detect_column_consistency())

    # Step 3: Detect data-related anomalies
    anomalies.extend(csv_stacker.detect_data_anomalies())

    # If no anomalies are detected, proceed with stacking and saving
    if not anomalies:
        anomalies.extend(csv_stacker.stack_dataframes())
        anomalies.extend(csv_stacker.save_as_pickle(output_pickle_file))
        logger.info("No anomalies detected. Process completed successfully.")
    else:
        logger.error("Anomalies detected; skipping stacking and saving.")

    return anomalies  # Return list of all detected anomalies

if __name__ == "__main__":
    detected_anomalies = stack_csvs_to_pickle()
    if detected_anomalies:
        logger.error(f"Detected Anomalies: {detected_anomalies}")
    else:
        logger.info("No anomalies detected. Process completed successfully.")
