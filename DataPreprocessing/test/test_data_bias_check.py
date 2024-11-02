#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytest
import pandas as pd
from src.data_bias_check_final import PM25Analysis  # Assuming the class is saved in a module
import tempfile

@pytest.fixture
def pm25_analysis():
    # Mock setup with a sample dataset for testing
    sample_data = {
        'parameter': ['pm25', 'pm25', 'pm25'],
        'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'value': [10, 20, 15],
        'month': [1, 2, 3],
    }
    df = pd.DataFrame(sample_data)
    analysis = PM25Analysis(df)
    return analysis

def test_load_and_filter_data(pm25_analysis):
    data = pm25_analysis.load_and_filter_data()
    assert not data.empty, "Data should not be empty after loading and filtering"
    assert all(data['parameter'] == 'pm25'), "Filtered data should only contain PM2.5 measurements"

def test_preprocess_data(pm25_analysis):
    pm25_analysis.load_and_filter_data()  # Ensure data is loaded
    preprocessed_data = pm25_analysis.preprocess_data()
    assert 'hour' in preprocessed_data.columns, "Preprocessed data should contain 'hour' column"
    assert 'day_of_week' in preprocessed_data.columns, "Preprocessed data should contain 'day_of_week' column"

def test_fairlearn_bias_check_with_flag(pm25_analysis):
    pm25_analysis.load_and_filter_data()
    pm25_analysis.preprocess_data()
    bias_df = pm25_analysis.fairlearn_bias_check_with_flag(sensitive_feature='month')
    assert 'bias_flag' in bias_df.columns, "Bias DataFrame should contain 'bias_flag' column"
    assert not bias_df.empty, "Bias DataFrame should not be empty"

def test_resample_biased_months(pm25_analysis):
    pm25_analysis.load_and_filter_data()
    pm25_analysis.preprocess_data()
    pm25_analysis.fairlearn_bias_check_with_flag(sensitive_feature='month')
    resampled_data = pm25_analysis.resample_biased_months()
    assert not resampled_data.empty, "Resampled data should not be empty"
    assert len(resampled_data) >= len(pm25_analysis.data_pm25), "Resampled data should have at least as many rows as the original data"




