import pandas as pd
import os
import json
import re
import logging
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store

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
MLMD_DB_PATH = os.path.join(os.getcwd(), 'mlmd_metadata_schema_original_dataset.db')

# 1. Set up the ML Metadata store
def setup_mlmd_store(db_path):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = db_path
    connection_config.sqlite.connection_mode = metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
    logger.info("ML Metadata store set up with SQLite database at %s", db_path)
    return metadata_store.MetadataStore(connection_config)

# 2. Define the schema and save it as a JSON file
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

# 3. Track schema and dataset as artifacts in MLMD
def track_artifacts(store, schema_path, dataset_path):
    # Register schema artifact type
    schema_type = metadata_store_pb2.ArtifactType()
    schema_type.name = 'Schema'
    schema_type.properties['version'] = metadata_store_pb2.INT
    schema_type.properties['file_path'] = metadata_store_pb2.STRING
    schema_type_id = store.put_artifact_type(schema_type)
    
    # Track schema artifact
    schema_artifact = metadata_store_pb2.Artifact()
    schema_artifact.type_id = schema_type_id
    schema_artifact.uri = schema_path
    schema_artifact.properties['version'].int_value = 1
    schema_artifact_id = store.put_artifacts([schema_artifact])
    logger.info("Schema artifact tracked with ID: %s", schema_artifact_id)

    # Register dataset artifact type
    dataset_type = metadata_store_pb2.ArtifactType()
    dataset_type.name = 'Dataset'
    dataset_type.properties['version'] = metadata_store_pb2.INT
    dataset_type.properties['file_path'] = metadata_store_pb2.STRING
    dataset_type_id = store.put_artifact_type(dataset_type)

    # Track dataset artifact
    dataset_artifact = metadata_store_pb2.Artifact()
    dataset_artifact.type_id = dataset_type_id
    dataset_artifact.uri = dataset_path
    dataset_artifact.properties['version'].int_value = 1
    dataset_artifact_id = store.put_artifacts([dataset_artifact])
    logger.info("Dataset artifact tracked with ID: %s", dataset_artifact_id)

# 4. Validate the dataset against the schema
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
    # Step 1: Set up MLMD store
    store = setup_mlmd_store(MLMD_DB_PATH)
    
    # Step 2: Define and save the schema
    define_and_save_schema(SCHEMA_FILE_PATH)
    
    # Step 3: Track artifacts in MLMD
    track_artifacts(store, SCHEMA_FILE_PATH, DATASET_FILE_PATH)
    
    # Step 4: Load and validate dataset
    is_valid = load_and_validate_dataset(DATASET_FILE_PATH, SCHEMA_FILE_PATH)
    if is_valid:
        logger.info("Data validation passed.")
    else:
        logger.error("Data validation failed.")

# Run the main function
if __name__ == "__main__":
    main_check_schema_original()
