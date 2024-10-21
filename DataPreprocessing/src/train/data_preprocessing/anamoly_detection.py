import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(f"Data loaded from {self.file_path}.")
    
    def handle_outliers(self, column_name='pm25'):
        if column_name not in self.data.columns:
            raise ValueError(f"'{column_name}' column not found in the DataFrame.")
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_median = self.data[column_name].median()
        self.data.loc[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound), column_name] = column_median
        print(f"Outliers in '{column_name}' replaced with median value {column_median}.")

    def replace_negative_with_zero(self, column_name='pm25'):
        if column_name not in self.data.columns:
            raise ValueError(f"'{column_name}' column not found in the DataFrame.")
        self.data[column_name] = self.data[column_name].clip(lower=0)
        print(f"Negative values in '{column_name}' replaced with 0.")

    def save_as_pickle(self, output_path):
        if self.data is not None:
            self.data.to_pickle(output_path)
            print(f"Cleaned DataFrame saved as '{output_path}'.")
        else:
            print("No data available to save. Please load and clean the data first.")

def main():
    # Path to the input pickle file and output pickle file
    file_path =  os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/train_data/no_null_train_data.pkl")
    output_pickle_file =os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/train_data/no_anamoly_train_data.pkl")
    cleaner = DataCleaner(file_path)
    cleaner.load_data()
    cleaner.handle_outliers(column_name='pm25')
    cleaner.replace_negative_with_zero(column_name='pm25')
    cleaner.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    main()
