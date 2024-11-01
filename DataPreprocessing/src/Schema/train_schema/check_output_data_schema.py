import pandas as pd
import numpy as np
import logging
import os
import json

# Set up logging configuration
def setup_logging():
    log_file_path = 'check_output_data_schema_train.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Constants for file paths
SCHEMA_FILE_PATH = os.path.join(os.getcwd(), 'data_schema_train.json')
STATS_FILE_PATH = os.path.join(os.getcwd(), 'data_statistics_train.json')
DATASET_FILE_PATH = os.path.join(os.getcwd(), 'dags/DataPreprocessing/src/data_store_pkl_files/train_data/feature_eng_train_data.pkl')

# Function to generate schema dynamically based on dataset
def generate_schema(data, required_columns=[]):
    schema = {}
    for column_name, dtype in data.dtypes.items():
        column_info = {
            "type": dtype.name,
            "required": column_name in required_columns or not data[column_name].isnull().any()
        }

        # Specific constraints for certain columns
        if column_name.startswith("pm25"):
            column_info["sign"] = "positive"
        elif "lag" in column_name or "rolling" in column_name or "ema" in column_name:
            column_info["sign"] = "positive"
        elif column_name == "hour":
            column_info.update({"min": 0, "max": 23})
        elif column_name == "day_of_week":
            column_info.update({"min": 0, "max": 6})
        elif column_name == "day_of_year":
            column_info.update({"min": 1, "max": 365})
        elif column_name == "month":
            column_info.update({"min": 1, "max": 12})
        elif "sin" in column_name or "cos" in column_name:
            column_info.update({"min": -1.0, "max": 1.0})
        
        schema[column_name] = column_info

    # Return the generated schema
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
            "unique_values": data[column].nunique()
        }
    
    # Return the generated statistics
    return stats

# Helper function to save data (schema or statistics) to a JSON file
def save_to_file(data, file_path):
    def convert_types(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    # Convert data for JSON serialization
    data = convert_types(data)
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")

# Validation function for each column based on the generated schema
def validate_column(column_name, column_data, column_schema):
    issues = []
    if column_schema.get("required", False) and column_data.isnull().any():
        issues.append(f"Missing values found in required column '{column_name}'")
    
    if column_schema.get("sign") == "positive" and (column_data < 0).any():
        issues.append(f"Negative values found in column '{column_name}' expected to be positive")
    
    if "min" in column_schema and column_data.min() < column_schema["min"]:
        issues.append(f"Column '{column_name}' has values below expected minimum {column_schema['min']}")
    if "max" in column_schema and column_data.max() > column_schema["max"]:
        issues.append(f"Column '{column_name}' has values above expected maximum {column_schema['max']}")
    
    if issues:
        for issue in issues:
            logger.warning(issue)
    return issues

# Validate the dataset against the dynamically generated schema
def validate_data(data, schema):
    has_pm25_boxcox = "pm25_boxcox" in data.columns
    has_pm25_log = "pm25_log" in data.columns

    if has_pm25_boxcox and has_pm25_log:
        logger.error("Both 'pm25_boxcox' and 'pm25_log' columns are present. Only one should be present.")
        return False
    elif not has_pm25_boxcox and not has_pm25_log:
        logger.error("Neither 'pm25_boxcox' nor 'pm25_log' column is present. One of them must be present.")
        return False

    all_issues = []
    for column_name, column_schema in schema.items():
        if column_name not in data.columns:
            if column_schema.get("required", False):
                error_msg = f"Missing required column '{column_name}' specified in the schema."
                logger.error(error_msg)
                all_issues.append(error_msg)
        else:
            column_data = data[column_name]
            all_issues.extend(validate_column(column_name, column_data, column_schema))
    
    if all_issues:
        logger.info("Validation issues summary:")
        for issue in all_issues:
            logger.info(issue)
    
    return not all_issues

# Main function to load data, generate schema and statistics, and run validation
def main_generate_schema_and_statistics():
    try:
        data = pd.read_pickle(DATASET_FILE_PATH)
        logger.info(f"Loaded dataset from {DATASET_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {DATASET_FILE_PATH}: {e}")
        return

    # Generate schema and statistics
    schema = generate_schema(data, required_columns=["pm25", "hour", "day_of_week", "month"])
    stats = generate_statistics(data)

    # Save schema and statistics to files
    save_to_file(schema, SCHEMA_FILE_PATH)
    save_to_file(stats, STATS_FILE_PATH)

    # Run data validation
    if validate_data(data, schema):
        logger.info("Data validation passed.")
    else:
        logger.info("Data validation failed.")

# Run the main function
if __name__ == "__main__":
    main_generate_schema_and_statistics()
