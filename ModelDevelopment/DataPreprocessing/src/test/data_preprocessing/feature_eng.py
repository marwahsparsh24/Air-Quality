import pandas as pd
import numpy as np
from scipy import stats
import os

class DataFeatureEngineer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.fitting_lambda = None
    
    def get_lambda(self):
        return self.fitting_lambda

    def load_data(self):
        self.data = pd.read_pickle(self.file_path)
        print(f"Data loaded from {self.file_path}.")
    
    def handle_skewness(self, column_name='pm25'):
        skewness = self.data[column_name].skew()
        print(f'Original Skewness: {skewness}')
        if np.abs(skewness)<0.5:
            return column_name
        else:
            self.data[f'{column_name}_log'] = np.log1p(self.data[column_name])
            log_skewness = self.data[f'{column_name}_log'].skew()
            print(f'Log Transformed Skewness: {log_skewness}')

            self.data[f'{column_name}_boxcox'], self.fitting_lambda = stats.boxcox(self.data[column_name] + 1)
            boxcox_skewness = self.data[f'{column_name}_boxcox'].skew()
            print(f'Box-Cox Transformed Skewness: {boxcox_skewness}')

            if abs(boxcox_skewness) < abs(log_skewness):
                self.data.drop(columns=[f'{column_name}_log'],inplace=True)
                print("Choosing Box-Cox transformed column.")
                return f'{column_name}_boxcox'
            else:
                print("Choosing Log transformed column.")
                self.data.drop(columns=[f'{column_name}_boxcox'],inplace=True)
                return f'{column_name}_log'

    def feature_engineering(self, chosen_column):
        # Create lag features
        for lag in range(1, 6):  # Creates lag_1 to lag_5
            self.data[f'lag_{lag}'] = self.data[chosen_column].shift(lag)

        self.data['rolling_mean_3'] = self.data[chosen_column].rolling(window=3).mean()
        self.data['rolling_mean_6'] = self.data[chosen_column].rolling(window=6).mean()
        self.data['rolling_mean_24'] = self.data[chosen_column].rolling(window=24).mean()
        self.data['rolling_std_3'] = self.data[chosen_column].rolling(window=3).std()
        self.data['rolling_std_6'] = self.data[chosen_column].rolling(window=6).std()
        self.data['rolling_std_24'] = self.data[chosen_column].rolling(window=24).std()
        self.data['ema_3'] = self.data[chosen_column].ewm(span=3, adjust=False).mean()
        self.data['ema_6'] = self.data[chosen_column].ewm(span=6, adjust=False).mean()
        self.data['ema_24'] = self.data[chosen_column].ewm(span=24, adjust=False).mean()
        self.data['diff_1'] = self.data[chosen_column].diff(1)
        self.data['diff_2'] = self.data[chosen_column].diff(2)
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['month'] = self.data.index.month
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['sin_day_of_week'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['cos_day_of_week'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data.dropna(inplace=True)
        print("Feature engineering completed and NaN values dropped.")

    def save_as_pickle(self, output_path):
        if self.data is not None:
            self.data.to_pickle(output_path)
            print(f"Processed DataFrame saved as '{output_path}'.")
        else:
            print("No data available to save. Please load and process the data first.")

def feature_engineering():
    # Path to the input pickle file and output pickle file
    file_path = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/test_data/no_anamoly_test_data.pkl")
    output_pickle_file = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/test_data/feature_eng_test_data.pkl")
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    engineer.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    feature_engineering()
