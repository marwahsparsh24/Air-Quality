import pandas as pd
import os
import json
import re
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
SCHEMA_FILE_PATH = os.path.join(os.getcwd(), 'custom_schema_original_dataset.json')
DATASET_FILE_PATH = os.path.join(os.getcwd(), 'DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl')


#Define the schema and save it as a JSON file
def define_and_save_schema(file_path):
    schema = {
        "columns": [
            {"name": "date", "type": "string", "required": True, "format": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"},
            {"name": "parameter", "type": "string", "required": True, "allowed_values": ["pm25", "co", "no", "no2", "o3", "so2", "pm10"]}
        ]
    }
    with open(file_path, 'w') as f:
        json.dump(schema, f)
    logger.info("Schema saved to %s", file_path)

def validate_data(data, schema):
    # Check if 'parameter' column contains 'pm25'
    parameter_col = data['parameter'] if 'parameter' in data.columns else None
    if parameter_col is None or not parameter_col.str.contains('pm25').any():
        logger.error("Error: 'parameter' column must contain at least one instance of 'pm25'.")
        return False
    
    # Validate 'date' format using regex
    date_format_regex = schema['columns'][0]['format']
    date_col = data['date'] if 'date' in data.columns else None
    if date_col is None or not date_col.astype(str).str.match(date_format_regex).all():
        logger.error("Error: Some entries in 'date' column do not match the required format %s", date_format_regex)
        return False

    logger.info("Dataset is valid according to the schema.")
    return True

# 5. Load the dataset and validate it
def load_and_validate_dataset(file_path, schema_path):
    # Load dataset
    data = pd.read_pickle(file_path)
    logger.info("Loaded dataset from %s", file_path)
    
    # Load schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    logger.info("Loaded schema from %s", schema_path)
    
    # Validate data
    is_valid = validate_data(data, schema)
    return is_valid

# Main function to orchestrate everything
def main_check_schema_original():
    # Define and save the schema
    define_and_save_schema(SCHEMA_FILE_PATH)
    
    # Step 4: Load and validate dataset
    is_valid = load_and_validate_dataset(DATASET_FILE_PATH, SCHEMA_FILE_PATH)
    if is_valid:
        logger.info("Data validation passed.")
    else:
        logger.error("Data validation failed.")

# Run the main function
if __name__ == "__main__":
    main_check_schema_original()
