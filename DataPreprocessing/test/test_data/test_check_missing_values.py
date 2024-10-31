import pytest
import pandas as pd
import numpy as np
import os
from src.train.data_preprocessing.check_missing_values import DataProcessor 

@pytest.fixture
def mock_data(tmp_path):
    print("Setting up mock_data fixture")
    data = pd.DataFrame({
        'pm25': [10, np.nan, 15, np.nan, 20],
        'other_column': [1, 2, 3, 4, 5]
    })
    file_path = tmp_path / "mock_data.pkl"
    data.to_pickle(file_path)
    return file_path 

@pytest.fixture
def processor(mock_data):
    # Initialize the DataProcessor with the file path of the mock data
    print("Setting up processor fixture")
    return DataProcessor(mock_data)

def test_load_data(processor):
    print("Running test_load_data")
    processor.load_data()
    assert processor.data is not None
    assert 'pm25' in processor.data.columns

def test_handle_missing_values(processor):
    print("Running test_handle_missing_values")
    processor.load_data()
    initial_nan_count = processor.data['pm25'].isna().sum()
    assert initial_nan_count > 0  # Check there are missing values initially

    processor.handle_missing_values()
    processed_nan_count = processor.data['pm25'].isna().sum()
    assert processed_nan_count == 0  # Ensure no missing values remain after processing

def test_save_as_pickle(processor, tmp_path):
    print("Running test_save_as_pickle")
    processor.load_data()
    processor.handle_missing_values()

    output_path = tmp_path / "processed_data.pkl"
    processor.save_as_pickle(output_path)

    assert output_path.exists()
    saved_data = pd.read_pickle(output_path)
    assert saved_data.equals(processor.data)  # Check if the saved data matches the processed data
