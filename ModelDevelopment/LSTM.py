import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path, time_steps=5):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.time_steps = time_steps
        self.model = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
        self.scaler_X = MinMaxScaler()

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

        # MinMax Scaling for X only
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)

    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.time_steps):
            Xs.append(X[i:(i + self.time_steps)])
            ys.append(y[i + self.time_steps])
        return np.array(Xs), np.array(ys)

    def build_model(self):
        # Define LSTM Model
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, activation='relu', input_shape=(self.time_steps, self.X_train_scaled.shape[1])),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        # Prepare sequences
        X_train_seq, y_train_seq = self.create_sequences(self.X_train_scaled, self.y_train.values)
        X_test_seq, y_test_seq = self.create_sequences(self.X_test_scaled, self.y_test.values)

        # Train the model
        self.model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=64, validation_data=(X_test_seq, y_test_seq))

    def evaluate_model(self):
        X_test_seq, _ = self.create_sequences(self.X_test_scaled, self.y_test.values)

        # Make predictions
        y_pred_boxcox = self.model.predict(X_test_seq)

        # Inverse Box-Cox transformation to get predictions back to the original PM2.5 scale
        y_pred_original = inv_boxcox(y_pred_boxcox.flatten(), self.lambda_value)

        # Evaluate the model on the original PM2.5 scale
        rmse_original = mean_squared_error(self.y_test_original[self.time_steps:], y_pred_original, squared=False)
        print(f"RMSE (Original PM2.5 target): {rmse_original}")

        return y_pred_original

    def save_weights(self):
        model_save_path = self.model_save_path
        with open(model_save_path, 'wb') as f:
            pd.to_pickle(self.model, f)
        print(f"Model saved at {model_save_path}")
    
    def load_weights(self):
        # Load the model weights
        self.build_model()  # Ensure the model is built before loading weights
        model_save_path = self.model_save_path
        with open(model_save_path, 'rb') as f:
            self.model = pd.read_pickle(f)
        print(f"Model loaded from {model_save_path}")

    def plot_results(self, y_pred_original):
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test_original.values, label='Actual PM2.5', color='blue')
        plt.plot(y_pred_original, label='Predicted PM2.5', color='red')
        plt.title('Actual vs Predicted PM2.5 Values')
        plt.xlabel('Time Step')
        plt.ylabel('PM2.5')
        plt.legend()

        # Save the plot
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_LSTM.png')
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")

# Main function to orchestrate the workflow
def main():
    curr_dir = os.getcwd()
    main_path = os.path.abspath(os.path.join(curr_dir, '.'))
    data_prepocessing_path_pkl = os.path.abspath(os.path.join(main_path, 'DataPreprocessing/src/data_store_pkl_files'))

    # Load Data using DataFeatureEngineer
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    sys.path.append(main_path)
    sys.path.append(os.path.abspath(os.path.join(main_path, 'DataPreprocessing')))
    sys.path.append(data_prepocessing_path_pkl)
    from DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer

    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()

    # Define file paths
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    test_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
    model_save_path = os.path.join(main_path, 'weights/lstm_pm25_model.pth')

    # Initialize the LSTMPM25Model class
    lstm_model = LSTMPM25Model(train_file, test_file, fitting_lambda, model_save_path)

    # Load data, train the model, and evaluate it
    lstm_model.load_data()
    lstm_model.build_model()
    lstm_model.train_model()
    y_pred_original = lstm_model.evaluate_model()

    # Save the model weights
    lstm_model.save_weights()

    # Load the model weights for future prediction
    lstm_model.load_weights()

    # Plot the results
    lstm_model.plot_results(y_pred_original)

if __name__ == "__main__":
    main()
