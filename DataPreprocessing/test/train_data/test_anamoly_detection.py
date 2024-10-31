import pytest
import pandas as pd
import numpy as np
import os
import sys
# sys.path.insert(0, '/Users/donyadabiri/Documents/PhD Northeastern/MLOps/Air-Quality/DataPreprocessing/src')
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# print(" debugging \n".join(sys.path))  # Debugging line to print `sys.path`
from src.train.data_preprocessing.anamoly_detection import DataCleaner


# Fixture for setting up and tearing down the DataCleaner instance
@pytest.fixture
def setup_cleaner(tmp_path):
    df = pd.DataFrame({
        'pm25': [10, 20, 30, 1000, -10, 25, 50, 75, 1200],
        'other_column': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    test_file_path = tmp_path / 'test_data.pkl'
    df.to_pickle(test_file_path)

    # Initialize DataCleaner with the test file path
    cleaner = DataCleaner(str(test_file_path))
    cleaner.load_data()
    print("setup_cleaner worked")
    return cleaner, df, test_file_path

def test_load_data(setup_cleaner):
    cleaner, df, _ = setup_cleaner
    assert cleaner.data is not None
    assert len(cleaner.data) == len(df)
    print("Test load_data passed.")

def test_handle_outliers(setup_cleaner):
    cleaner, df, _ = setup_cleaner
    median = df['pm25'].median()

    # Run outlier handling
    cleaner.handle_outliers(column_name='pm25')

    # Check that extreme values are replaced with the median
    assert cleaner.data.loc[3, 'pm25'] == median  # 1000 should be replaced
    assert cleaner.data.loc[8, 'pm25'] == median  # 1200 should be replaced
    print("Test handle_outliers passed.")

def test_replace_negative_with_zero(setup_cleaner):
    cleaner, _, _ = setup_cleaner

    # Run negative value replacement
    cleaner.replace_negative_with_zero(column_name='pm25')

    # Check that negative values are replaced with zero
    assert (cleaner.data['pm25'] >= 0).all()
    assert cleaner.data.loc[4, 'pm25'] == 0  # -10 should be replaced with 0
    print("Test replace_negative_with_zero passed.")

# def test_column_not_found_error(setup_cleaner):
#     cleaner, _, _ = setup_cleaner

#     # Test for non-existing column in outlier handling
#     with pytest.raises(ValueError, match="'non_existent_column' column not found in the DataFrame."):
#         cleaner.handle_outliers(column_name='non_existent_column')

#     # Test for non-existing column in negative replacement
#     with pytest.raises(ValueError, match="'non_existent_column' column not found in the DataFrame."):
#         cleaner.replace_negative_with_zero(column_name='non_existent_column')

#     print("Test column_not_found_error passed.")

def test_save_as_pickle(setup_cleaner, tmp_path):
    cleaner, _, _ = setup_cleaner
    output_path = tmp_path / 'cleaned_data.pkl'

    # Run the save method
    cleaner.save_as_pickle(str(output_path))

    # Check if the file was created
    assert os.path.exists(output_path)

    # Load the saved file and compare the content
    saved_data = pd.read_pickle(output_path)
    pd.testing.assert_frame_equal(saved_data, cleaner.data)
    print("Test save_as_pickle passed.")
