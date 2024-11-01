import pandas as pd
import numpy as np
import os
import json
import logging

# Set up logging configuration to log to a file and console
log_file_path = os.path.join(os.getcwd(), 'process_check_schema_air_pollution.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),   # Log to a file
        logging.StreamHandler()               # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Constants for file paths
SCHEMA_FILE_PATH = os.path.join(os.getcwd(), 'dags/custom_schema_generated_from_api.json')
DATASET_FILE_PATH = os.path.join(os.getcwd(), 'dags/DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl')
STATS_FILE_PATH = os.path.join(os.getcwd(), 'dags/air_pollution_stats.json')

# Function to generate schema based on dataset structure
def generate_schema(data):
    print(data.dtypes.items())
    # Check if 'date' and 'parameter' columns are present
    required_columns = ['date', 'parameter']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    # Schema generation based on data structure
    schema = {"columns": []}
    for column_name, dtype in data.dtypes.items():
        column_info = {
            "name": column_name,
            "type": dtype.name,
            "required": not data[column_name].isnull().any()  # True if no missing values
        }

        # Adding specific constraints (e.g., date format for "date" column, allowed values for "parameter" column)
        if column_name == "date":
            column_info["format"] = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"
        elif column_name == "parameter":
            column_info["allowed_values"] = list(data[column_name].unique())
        
        schema["columns"].append(column_info)
    
    return schema

# Function to generate statistics for each column
def generate_statistics(data):
    stats = {}
    for column in data.columns:
        stats[column] = {
            "mean": data[column].mean() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "std_dev": data[column].std() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "min": data[column].min(),
            "max": data[column].max(),
            "missing_values": data[column].isnull().sum(),
            "unique_values": len(data[column].unique())
        }
    return stats

# Save data (schema or statistics) to a JSON file
def save_to_file(data, file_path):
    def convert_types(obj):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    # Convert the data for serialization
    data = convert_types(data)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info("Data saved to %s", file_path)

# Load the dataset, generate schema and statistics, and save them
def main_generate_schema_and_statistics():
    data = pd.read_pickle(DATASET_FILE_PATH)
    logger.info("Loaded dataset from %s", DATASET_FILE_PATH)

    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Loaded data is not a DataFrame or Series.")

    # Generate schema and statistics
    schema = generate_schema(data)
    stats = generate_statistics(data)

    # Save schema and statistics
    save_to_file(schema, SCHEMA_FILE_PATH)
    save_to_file(stats, STATS_FILE_PATH)

# Run the main function
if __name__ == "__main__":
    main_generate_schema_and_statistics()
