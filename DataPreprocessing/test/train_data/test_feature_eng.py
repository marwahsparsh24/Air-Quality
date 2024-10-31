import pytest
import pandas as pd
import numpy as np
import os
from src.train.data_preprocessing.feature_eng import DataFeatureEngineer

@pytest.fixture
def mock_data(tmp_path):
    data = pd.DataFrame({
        'pm25': [10, 20, 30, 1000, 15, 25, 35, 40, 50, 60],
        'other_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }, index=pd.date_range("2023-01-01", periods=10, freq="h"))
    
    file_path = tmp_path / "mock_data.pkl"
    data.to_pickle(file_path)
    return file_path

@pytest.fixture
def data_engineer(mock_data):
    #Initializes the DataFeatureEngineer with the mock data file.
    engineer = DataFeatureEngineer(file_path=mock_data)
    engineer.load_data()
    return engineer

def test_load_data(data_engineer):
    #Tests loading data from a file.
    assert data_engineer.data is not None
    assert 'pm25' in data_engineer.data.columns

def test_handle_skewness(data_engineer):
    #Tests skewness handling with log and Box-Cox transformations.
    chosen_column = data_engineer.handle_skewness(column_name='pm25')
    
    #Verify that a transformation column was created
    assert chosen_column in data_engineer.data.columns
    assert data_engineer.fitting_lambda is not None or 'pm25_log' in data_engineer.data.columns

def test_feature_engineering(data_engineer):
    #Tests if feature engineering adds the expected columns.
    chosen_column = data_engineer.handle_skewness(column_name='pm25')
    data_engineer.feature_engineering(chosen_column)
    
    #Verify lag features
    for lag in range(1, 6):
        assert f'lag_{lag}' in data_engineer.data.columns
    
    #Verify rolling statistics
    assert 'rolling_mean_3' in data_engineer.data.columns
    assert 'rolling_mean_6' in data_engineer.data.columns
    assert 'rolling_std_3' in data_engineer.data.columns
    assert 'rolling_std_6' in data_engineer.data.columns
    
    #Check that NaN values were dropped
    assert not data_engineer.data.isnull().values.any()

def test_save_as_pickle(data_engineer, tmp_path):
    #Tests saving the DataFrame as a pickle file.
    chosen_column = data_engineer.handle_skewness(column_name='pm25')
    data_engineer.feature_engineering(chosen_column)

    output_path = tmp_path / "processed_data.pkl"
    data_engineer.save_as_pickle(output_path)
    
    #Verify the file was created
    assert os.path.exists(output_path)
    
    #Check if the saved data matches the processed data
    loaded_data = pd.read_pickle(output_path)
    pd.testing.assert_frame_equal(loaded_data, data_engineer.data)

