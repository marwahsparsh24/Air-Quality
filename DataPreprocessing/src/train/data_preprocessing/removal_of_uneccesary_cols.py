import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(f"Data loaded from {self.file_path}.")
    
    def drop_columns(self, columns_to_drop):
        if self.data is not None:
            columns_present = [col for col in columns_to_drop if col in self.data.columns]
        
            if columns_present:
                self.data.drop(columns=columns_present, inplace=True)
                print(f"Successfully dropped columns: {columns_present}")
            else:
                print(f"None of the columns {columns_to_drop} exist in the DataFrame.")
        self.data.set_index('date',inplace=True)
        
    def save_as_pickle(self, output_path):
        if self.data is not None:
            self.data.to_pickle(output_path)
            print(f"Cleaned DataFrame saved as '{output_path}'.")
        else:
            print("No data available to save. Please load and clean the data first.")

def remove_uneccesary_cols():
    # Path to the input pickle file and output pickle file
    file_path = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/train_data/pivoted_train_data.pkl")
    output_pickle_file =  os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/train_data/cleaned_train_data.pkl")
    columns_to_drop = ['co', 'no', 'no2', 'o3', 'so2']
    cleaner = DataCleaner(file_path)
    cleaner.load_data()
    cleaner.drop_columns(columns_to_drop)
    cleaner.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    remove_uneccesary_cols()
