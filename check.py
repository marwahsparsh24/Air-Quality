import pickle5 as pickle
import pandas as pd
with open('processed-train-feature_eng_data.pkl', 'rb') as file:
    data = pickle.load(file)
print(data.head())
print(data['timestamp'])
 
 

 
# # Save the DataFrame to a CSV file
# data.to_csv('new_data_org.csv', index=False)