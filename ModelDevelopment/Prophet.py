# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from scipy.special import inv_boxcox
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import os
# import sys

# # Load data and features using your DataFeatureEngineer class
# curr_dir = os.getcwd()
# main_path = os.path.abspath(os.path.join(curr_dir, '.'))
# data_prepocessing_path = os.path.abspath(os.path.join(main_path, 'DataPreprocessing'))
# data_prepocessing_path_pkl = os.path.abspath(os.path.join(data_prepocessing_path, 'src/data_store_pkl_files'))
# sys.path.append(main_path)
# sys.path.append(data_prepocessing_path)
# sys.path.append(data_prepocessing_path_pkl)

# from DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer

# # Load and prepare data
# file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
# output_pickle_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
# engineer = DataFeatureEngineer(file_path)
# engineer.load_data()
# chosen_column = engineer.handle_skewness(column_name='pm25')
# engineer.feature_engineering(chosen_column)
# fitting_lambda = engineer.get_lambda()

# train_data = pd.read_pickle(os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl'))
# for column in train_data.columns:
#     if column == 'pm25_boxcox' or column == 'pm25_log':
#         y_train = train_data[column]
#         break
# y_train_original = train_data['pm25']  # Original data for evaluation
# X_train = train_data.drop(columns=['pm25'])

# test_data = pd.read_pickle(os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl'))
# for column in test_data.columns:
#     if column == 'pm25_boxcox' or column == 'pm25_log':
#         y_test = test_data[column]
#         break
# y_test_original = test_data['pm25']
# X_test = test_data.drop(columns=['pm25'])

# df_train = pd.DataFrame({
#     'ds': train_data.index.tz_localize(None),  # Remove timezone from 'ds' column
#     'y': y_train  # Box-Cox transformed target
# })

# # Initialize the Prophet model
# model = Prophet()

# # Fit the model on training data
# model.fit(df_train)

# # Create future dataframe (forecasting horizon)
# future = model.make_future_dataframe(periods=len(test_data))

# # Ensure no timezone in 'future'
# future['ds'] = future['ds'].dt.tz_localize(None)

# # Predict on the future dataframe
# forecast = model.predict(future)

# # Get predicted values for the Box-Cox transformed scale
# y_pred_boxcox = forecast['yhat'][-len(test_data):].values

# # Inverse Box-Cox transformation to get predictions back to the original PM2.5 scale
# y_pred_original = inv_boxcox(y_pred_boxcox, fitting_lambda)

# # Check for NaN values in the predicted values
# print(f"Number of NaN values in predictions: {np.isnan(y_pred_original).sum()}")
# print(f"Number of NaN values in true values: {np.isnan(y_test_original).sum()}")

# # Option 1: Remove rows with NaN values
# valid_idx = ~np.isnan(y_pred_original) & ~np.isnan(y_test_original)
# y_pred_original_valid = y_pred_original[valid_idx]
# y_test_original_valid = y_test_original[valid_idx]

# rmse_original = mean_squared_error(y_test_original_valid, y_pred_original_valid, squared=False)
# print(f"RMSE (Original PM2.5 target): {rmse_original}")

# plt.figure(figsize=(10, 6))
# plt.plot(y_test_original_valid.values, label='Actual PM2.5')
# plt.plot( y_pred_original_valid, label='Predicted PM2.5', color='red')
# plt.title('PM2.5 Forecast with LSTM')
# plt.xlabel('Time Step')
# plt.ylabel('PM2.5')
# plt.legend()
# plt.savefig(os.path.join(main_path,'artifacts/pm25_actual_vs_predicted_Prophet.png'))


import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle

class ProphetPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.model = Prophet()  # Initialize Prophet model
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
    def load_data(self):
        # Load training and test data
        train_data = pd.read_pickle(self.train_file)
        test_data = pd.read_pickle(self.test_file)

        # Extract Box-Cox transformed y and original y
        for column in train_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_train = train_data[column]
                break
        self.y_train_original = train_data['pm25']
        self.X_train = train_data.drop(columns=['pm25'])

        for column in test_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_test = test_data[column]
                break
        self.y_test_original = test_data['pm25']
        self.X_test = test_data.drop(columns=['pm25'])

    def preprocess_data(self):
        # Prepare training data in Prophet format
        self.df_train = pd.DataFrame({
            'ds': self.X_train.index.tz_localize(None),  # Remove timezone
            'y': self.y_train  # Use Box-Cox transformed target
        })

    def train_model(self):
        # Train the Prophet model
        self.model.fit(self.df_train)

    def evaluate(self):
        # Make future dataframe
        future = self.model.make_future_dataframe(periods=len(self.X_test))
        future['ds'] = future['ds'].dt.tz_localize(None)  # Remove timezone

        # Predict on the test data
        forecast = self.model.predict(future)
        y_pred_boxcox = forecast['yhat'][-len(self.X_test):].values

        # Inverse Box-Cox transformation to get predictions back to original PM2.5 scale
        y_pred_original = inv_boxcox(y_pred_boxcox, self.lambda_value)

        # Handle NaN values in predictions
        valid_idx = ~np.isnan(y_pred_original) & ~np.isnan(self.y_test_original)
        y_pred_original_valid = y_pred_original[valid_idx]
        y_test_original_valid = self.y_test_original[valid_idx]

        # Evaluate RMSE on the original PM2.5 scale
        rmse_original = mean_squared_error(y_test_original_valid, y_pred_original_valid, squared=False)
        print(f"RMSE (Original PM2.5 target): {rmse_original}")

        return y_pred_original_valid

    def save_weights(self):
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved at {self.model_save_path}")

    def load_weights(self):
        # Load the Prophet model from the specified path
        with open(self.model_save_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_save_path}")

    def plot_results(self, y_pred_original_valid):
        # Plot actual vs predicted values
        plt.figure(figsize=(10,6))
        plt.plot(self.y_test_original.values, label='Actual PM2.5', color='blue')
        plt.plot(y_pred_original_valid, label='Predicted PM2.5', color='red')
        plt.title('Actual vs Predicted PM2.5 Values with Prophet')
        plt.xlabel('Time')
        plt.ylabel('PM2.5 Value')
        plt.legend()

        # Save the plot as a PNG file
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_Prophet.png')
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")

# Main function to orchestrate the workflow
def main():
    curr_dir = os.getcwd()
    main_path = os.path.abspath(os.path.join(curr_dir, '.'))
    data_prepocessing_path = os.path.abspath(os.path.join(main_path, 'DataPreprocessing'))
    data_prepocessing_path_pkl = os.path.abspath(os.path.join(data_prepocessing_path, 'src/data_store_pkl_files'))

    # Step 1: Load Data using DataFeatureEngineer
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    sys.path.append(main_path)
    sys.path.append(data_prepocessing_path)
    sys.path.append(data_prepocessing_path_pkl)
    from DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer

    # Initialize DataFeatureEngineer to preprocess the data and fetch the lambda value
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()

    # Step 2: Define file paths
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    test_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
    model_save_path = os.path.join(main_path, 'weights/prophet_pm25_model.pth')

    # Step 3: Initialize the ProphetPM25Model class
    prophet_model = ProphetPM25Model(train_file, test_file, fitting_lambda, model_save_path)

    # Step 4: Load data, preprocess, train the model, and evaluate it
    prophet_model.load_data()
    prophet_model.preprocess_data()
    prophet_model.train_model()
    y_pred_original = prophet_model.evaluate()

    # Step 5: Save the model weights
    prophet_model.save_weights()

    # Step 6: Load the model weights for future prediction
    prophet_model.load_weights()

    # Step 7: Plot the results
    prophet_model.plot_results(y_pred_original)

if __name__ == "__main__":
    main()
