import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataSplitter:
    def __init__(self, pickle_file_path):
        self.pickle_file_path = pickle_file_path
        self.dataframe = None
        self.train_df = None
        self.test_df = None

    def load_pickle(self):
        self.dataframe = pd.read_pickle(self.pickle_file_path)
        print(f"Loaded data from {self.pickle_file_path}.")
    
    def split_data(self, test_size=0.2, random_state=42):
        if self.dataframe is None:
            print("No DataFrame to split. Please load the pickle file first.")
            return
        
        self.train_df, self.test_df = train_test_split(self.dataframe, test_size=test_size, random_state=random_state)
        print(f"Data split into training (size={len(self.train_df)}) and testing (size={len(self.test_df)}) sets.")
    
    def save_as_pickle(self, train_output_path, test_output_path):
        if self.train_df is None or self.test_df is None:
            print("No split data to save. Please split the data first.")
            return
        self.train_df.to_pickle(train_output_path)
        self.test_df.to_pickle(test_output_path)
        print(f"Training DataFrame saved as '{train_output_path}' and Testing DataFrame saved as '{test_output_path}'.")


def split():
    pickle_file_path = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/air_pollution.pkl")
    train_output_pickle_file = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/train_data/train_data.pkl")
    test_output_pickle_file =os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/test_data/test_data.pkl")
    data_splitter = DataSplitter(pickle_file_path)
    data_splitter.load_pickle()
    data_splitter.split_data(test_size=0.2, random_state=42)
    data_splitter.save_as_pickle(train_output_pickle_file, test_output_pickle_file)

if __name__ == "__main__":
    split()
