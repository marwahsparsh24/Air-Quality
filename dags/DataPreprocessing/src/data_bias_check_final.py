#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame
from sklearn.utils import resample
import os


class PM25Analysis:
    def __init__(self, data_source):
        if isinstance(data_source, str):
            self.filepath = data_source
            self.data_pm25 = None  # Data will be loaded from the file later
        elif isinstance(data_source, pd.DataFrame):
            self.data_pm25 = data_source  # Directly assign the DataFrame
            self.filepath = None  # No file path needed
        else:
            raise ValueError("data_source should be a file path or a DataFrame")

        self.results = {}

    def load_and_filter_data(self):
        if self.data_pm25 is None and self.filepath is not None:
            # Load data only if data_pm25 is None and filepath is provided
            data = pd.read_pickle(self.filepath)
            self.data_pm25 = data[data['parameter'] == 'pm25']
        elif self.data_pm25 is not None:
            # Filter if the DataFrame was provided directly
            self.data_pm25 = self.data_pm25[self.data_pm25['parameter'] == 'pm25']
        else:
            raise ValueError("No data source available.")
        return self.data_pm25
    
    # Method to preprocess data and add time-related columns
    def preprocess_data(self):
        self.data_pm25['date'] = pd.to_datetime(self.data_pm25['date'], errors='coerce')
        self.data_pm25['hour'] = self.data_pm25['date'].dt.hour
        self.data_pm25['day_of_week'] = self.data_pm25['date'].dt.day_name()
        self.data_pm25['month'] = self.data_pm25['date'].dt.month
        seasons = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                   9: 'Fall', 10: 'Fall', 11: 'Fall'}
        self.data_pm25['season'] = self.data_pm25['month'].map(seasons)
        print("Preprocessed Data with Time-related Columns:\n", self.data_pm25[['date', 'hour', 'day_of_week', 'month', 'season']].head())
        return self.data_pm25
    
    # Method to plot bias checks
    def plot_bias_checks(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data_pm25, x='hour')
        plt.title('Hourly Bias Check for PM2.5')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data_pm25, x='day_of_week', 
                      order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.title('Day of the Week Bias Check for PM2.5')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data_pm25, x='month')
        plt.title('Monthly Bias Check for PM2.5')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data_pm25, x='season', order=['Winter', 'Spring', 'Summer', 'Fall'])
        plt.title('Seasonal Bias Check for PM2.5')
        plt.show()

    # Fairlearn bias check based on a sensitive feature, showing only biased groups
    def fairlearn_bias_check_with_flag(self, sensitive_feature, threshold=0.2):
        if sensitive_feature not in self.data_pm25.columns:
            raise ValueError(f"{sensitive_feature} not found in data columns.")
        
        # Calculate the metric frame for counts grouped by the sensitive feature
        metric_frame = MetricFrame(
            metrics={'count': lambda x, y: len(x)},
            y_true=self.data_pm25['value'],
            y_pred=self.data_pm25['value'],  # Using y_true as y_pred
            sensitive_features=self.data_pm25[sensitive_feature]
        )
        
        # Calculate the average count across all groups
        avg_count = metric_frame.by_group['count'].mean()
        
        # Add a 'bias_flag' column to indicate if a group is biased
        biased_groups_df = metric_frame.by_group['count'].to_frame(name='count')
        biased_groups_df['bias_flag'] = ((biased_groups_df['count'] > avg_count * (1 + threshold)) |
                                         (biased_groups_df['count'] < avg_count * (1 - threshold)))
        
        # Store and print all groups with the bias flag
        self.results[f'fairlearn_bias_{sensitive_feature}_with_bias_flag'] = biased_groups_df
        print(f"All Groups with Bias Flag for {sensitive_feature} (Threshold={threshold}):\n", biased_groups_df)
        return biased_groups_df

    # Function to perform resampling on biased months and save the result
    def resample_biased_months(self, target_feature='month', save_path=None):
        if f'fairlearn_bias_{target_feature}_with_bias_flag' not in self.results:
            raise ValueError(f"Run fairlearn_bias_check_with_flag with {target_feature} before resampling.")
        
        biased_groups_df = self.results[f'fairlearn_bias_{target_feature}_with_bias_flag']
        
        # Calculate the target count based on the average count of unbiased months
        unbiased_counts = biased_groups_df[~biased_groups_df['bias_flag']]['count']
        target_count = int(unbiased_counts.mean())
        
        # Plot the distribution before resampling
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data_pm25, x=target_feature)
        plt.title(f"Original Distribution of {target_feature} Before Resampling")
        plt.show()

        # Separate data into biased and unbiased groups
        biased_months = biased_groups_df[biased_groups_df['bias_flag']].index
        biased_data = self.data_pm25[self.data_pm25[target_feature].isin(biased_months)]
        unbiased_data = self.data_pm25[~self.data_pm25[target_feature].isin(biased_months)]
        
        resampled_data = []
        
        for month in biased_months:
            month_data = biased_data[biased_data[target_feature] == month]
            if len(month_data) > target_count:
                # Downsample if count is higher than target
                month_data_resampled = resample(month_data, replace=False, n_samples=target_count, random_state=42)
            else:
                # Upsample if count is lower than target
                month_data_resampled = resample(month_data, replace=True, n_samples=target_count, random_state=42)
            resampled_data.append(month_data_resampled)
        
        # Concatenate resampled data with unbiased data
        resampled_data = pd.concat(resampled_data + [unbiased_data], ignore_index=True)
        
        # Plot the distribution after resampling
        plt.figure(figsize=(10, 6))
        sns.countplot(data=resampled_data, x=target_feature)
        plt.title(f"Resampled Distribution of {target_feature}")
        plt.show()
        
        print("Resampled Data based on biased months:\n", resampled_data.head())

        # Save the resampled data to a file if a path is provided
        if save_path:
            resampled_data.to_pickle(save_path)
            print(f"Resampled data saved to {save_path}")

        return resampled_data


# Main function to run the analysis and display the results
def main(filepath, sensitive_feature=None, threshold=0.2, save_path=None):
    analysis = PM25Analysis(filepath)
    analysis.load_and_filter_data()
    analysis.preprocess_data()
    analysis.plot_bias_checks()
    
    # Fairlearn bias check if a sensitive feature is provided
    if sensitive_feature:
        analysis.fairlearn_bias_check_with_flag(sensitive_feature, threshold)
        resampled_data = analysis.resample_biased_months(target_feature=sensitive_feature, save_path=save_path)
        print("Final Resampled Data:\n", resampled_data.head())

def bias_main():
    input_path = os.path.join(os.getcwd(),"dags/DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl")
    save_path = os.path.join(os.getcwd(),"dags/DataPreprocessing/src/data_store_pkl_files/resampled_data.pkl")
    # Run the main function with the specified file path and save path
    main(
       input_path, 
        sensitive_feature='month',
        threshold=0.2,
        save_path=save_path)

if __name__ == "__main__":
    bias_main()


