import pytest
import pandas as pd
import os
from src.train.data_preprocessing.removal_of_uneccesary_cols import DataCleaner

@pytest.fixture
def mock_data(tmp_path):
    data = pd.DataFrame({
        'date': pd.date_range("2023-01-01", periods=5, freq="D"),
        'pm25': [10, 15, 20, 25, 30],
        'co': [1, 2, 3, 4, 5],
        'no': [0.1, 0.2, 0.3, 0.4, 0.5],
        'o3': [0.05, 0.06, 0.07, 0.08, 0.09],
    })
    
    file_path = tmp_path / "mock_data.pkl"
    data.to_pickle(file_path)
    return file_path

@pytest.fixture
def data_cleaner(mock_data):
    cleaner = DataCleaner(file_path=mock_data)
    cleaner.load_data()
    return cleaner

def test_load_data(data_cleaner):
    assert data_cleaner.data is not None
    assert 'date' in data_cleaner.data.columns
    assert 'pm25' in data_cleaner.data.columns

def test_drop_columns(data_cleaner):
    columns_to_drop = ['co', 'no', 'so2']  # 'so2' does not exist in the data
    data_cleaner.drop_columns(columns_to_drop)

    #Check if columns that exist in the DataFrame were dropped
    assert 'co' not in data_cleaner.data.columns
    assert 'no' not in data_cleaner.data.columns
    
    #Ensure 'so2' didn't affect anything since it wasn't present
    assert 'pm25' in data_cleaner.data.columns
    assert 'o3' in data_cleaner.data.columns
    
    #Verify that 'date' was set as the index
    assert data_cleaner.data.index.name == 'date'

def test_save_as_pickle(data_cleaner, tmp_path):
    #Tests saving the cleaned data as a pickle file.
    columns_to_drop = ['co', 'no']  # Drop some columns before saving
    data_cleaner.drop_columns(columns_to_drop)
    
    output_path = tmp_path / "cleaned_data.pkl"
    data_cleaner.save_as_pickle(output_path)
    
    #Verify that the output file was created
    assert os.path.exists(output_path)
    
    #Load the saved data and check that it matches the cleaned data
    loaded_data = pd.read_pickle(output_path)
    pd.testing.assert_frame_equal(loaded_data, data_cleaner.data)
