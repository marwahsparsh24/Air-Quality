import pickle
with open('feature_eng_train_data.pkl', 'rb') as file:
    data = pickle.load(file)
 
 
import pandas as pd
 
# Save the DataFrame to a CSV file
data.to_csv('new_data_org.csv', index=False)