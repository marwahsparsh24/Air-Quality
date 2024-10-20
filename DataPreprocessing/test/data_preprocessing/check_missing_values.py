import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(self.data.head())
        print(f"Data loaded from {self.file_path}.")
    
    def handle_missing_values(self):
        if self.data.isnull().sum().any():  # Check if any missing values exist
            print("Missing values found. Handling missing data...")
            if 'pm25' in self.data.columns:
                self.data['pm25'] = self.data['pm25'].interpolate(method='linear')  # Apply linear interpolation
                print("'pm25' missing values interpolated.")
            else:
                print("'pm25' column not found for interpolation.")
        else:
            print("No missing values found.")

    def save_as_pickle(self, output_path):
        if self.data is not None:
            self.data.to_pickle(output_path)
            print(f"Processed DataFrame saved as '{output_path}'.")
        else:
            print("No data available to save. Please load and process the data first.")

def main():
    # Path to the input pickle file and output pickle file
    file_path = 'data_store_pkl_files/test_data/cleaned_test_data.pkl'
    output_pickle_file = 'data_store_pkl_files/test_data/no_null_test_data.pkl'
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.handle_missing_values()
    processor.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    main()
