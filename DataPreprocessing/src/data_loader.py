import pandas as pd
import os

class CSVStacker:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.dataframes = []
        self.stacked_df = None

    def load_csv_files(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        for file in csv_files:
            file_path = os.path.join(self.folder_path, file)
            df = pd.read_csv(file_path)
            self.dataframes.append(df)
        print(f"Loaded {len(csv_files)} CSV files.")
    
    def stack_dataframes(self):
        if not self.dataframes:
            print("No DataFrames to stack. Please load CSV files first.")
            return
        self.stacked_df = pd.concat(self.dataframes, ignore_index=True)
        print("DataFrames stacked successfully.")
    
    def save_as_pickle(self, output_path):
        if self.stacked_df is None:
            print("No DataFrame to save. Please stack the dataframes first.")
            return
        self.stacked_df.to_pickle(output_path)
        print(f"Stacked DataFrame saved as '{output_path}'.")


def stack_csvs_to_pickle():
    folder_path =os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/csv")
    output_pickle_file = os.path.join(os.getcwd(),"DataPreprocessing/src/data_store_pkl_files/resampled_data.pkl")
    csv_stacker = CSVStacker(folder_path)
    csv_stacker.load_csv_files()
    csv_stacker.stack_dataframes()
    csv_stacker.save_as_pickle(output_pickle_file)

if __name__ == "__main__":
    stack_csvs_to_pickle()
