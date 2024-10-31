import pytest
import pandas as pd
import numpy as np
import os
from src.train.data_preprocessing.pivoting_data import DataProcessor

@pytest.fixture
def mock_data(tmp_path):
    #Create a sample DataFrame with 'date', 'parameter', and 'value' columns
    data = pd.DataFrame({
        'date': pd.date_range("2023-01-01", periods=5, freq="D"),
        'parameter': ['pm25', 'pm25', 'pm10', 'pm10', 'pm25'],
        'value': [10, 15, 20, 25, 30]
    })
    
    file_path = tmp_path / "mock_data.pkl"
    data.to_pickle(file_path)
    return file_path

@pytest.fixture
def data_processor(mock_data):
    """Initializes the DataProcessor with the mock data file."""
    processor = DataProcessor(file_path=mock_data)
    processor.load_data()
    return processor

def test_load_data(data_processor):
    """Tests loading data from a file."""
    assert data_processor.data is not None
    assert 'date' in data_processor.data.columns
    assert 'parameter' in data_processor.data.columns
    assert 'value' in data_processor.data.columns

def test_process_dates(data_processor):
    #Tests if the date column is correctly converted to datetime.
    data_processor.process_dates()
    assert pd.api.types.is_datetime64_any_dtype(data_processor.data['date'])

def test_process_dates_missing_column(data_processor):
    #Tests that process_dates raises an error if the 'date' column is missing.
    data_processor.data.drop(columns=['date'], inplace=True)
    with pytest.raises(ValueError, match="No 'date' column in the DataFrame."):
        data_processor.process_dates()

def test_pivot_data(data_processor):
    #Tests if pivoting works correctly and produces the expected pivoted columns.
    data_processor.process_dates()
    data_processor.pivot_data()
    assert data_processor.pivoted_data is not None
    assert 'date' in data_processor.pivoted_data.columns
    assert 'pm25' in data_processor.pivoted_data.columns
    assert 'pm10' in data_processor.pivoted_data.columns

def test_pivot_data_missing_columns(data_processor):
    #Tests that pivot_data raises an error if required columns are missing.
    data_processor.data.drop(columns=['parameter'], inplace=True)
    with pytest.raises(ValueError, match="Missing one or more required columns: 'date', 'parameter', 'value'."):
        data_processor.pivot_data()

def test_save_as_pickle(data_processor, tmp_path):
    #Tests saving the pivoted data as a pickle file.
    data_processor.process_dates()
    data_processor.pivot_data()
    
    output_path = tmp_path / "pivoted_data.pkl"
    data_processor.save_as_pickle(output_path)
    
    #Verify the file was created
    assert os.path.exists(output_path)
    
    #Check if the saved data matches the processed data
    loaded_data = pd.read_pickle(output_path)
    pd.testing.assert_frame_equal(loaded_data, data_processor.pivoted_data)
