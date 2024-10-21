import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error
import os

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.pivoted_data = None

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(self.data.head())
        print(f"Data loaded from {self.file_path}")
    
    def process_dates(self):
        if 'date' not in self.data.columns:
            raise ValueError("No 'date' column in the DataFrame.")
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        print("Date column converted to datetime.")
    
    def pivot_data(self):
        if 'date' not in self.data.columns or 'parameter' not in self.data.columns or 'value' not in self.data.columns:
            raise ValueError("Missing one or more required columns: 'date', 'parameter', 'value'.")
        
        self.pivoted_data = self.data.pivot_table(index='date', columns='parameter', values='value').reset_index()
        print("Data pivoted successfully.")
    
    def save_as_pickle(self, output_path):
        if self.pivoted_data is not None:
            self.pivoted_data.to_pickle(output_path)
            print(f"Pivoted DataFrame saved as '{output_path}'.")
        else:
            print("No pivoted data to save. Please pivot the data first.")


def pivot_parameters():
    file_path = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/test_data/test_data.pkl")
    output_pickle_file = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/test_data/pivoted_test_data.pkl")
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.process_dates()
    processor.pivot_data()
    processor.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    pivot_parameters()
