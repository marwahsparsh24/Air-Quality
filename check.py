import pickle
with open('processed-train-feature_eng_data.pkl', 'rb') as file:
    data = pickle.load(file)
 
 
import pandas as pd
 
# Save the DataFrame to a CSV file
data.to_csv('new_data.csv', index=False)