import requests
import json
import pandas as pd
import os
from google.cloud import storage
import io
from io import BytesIO


api_key = "3fbb719154baaed8164cee7d57ba31903cdd19fdd7613e33ace632a0d851894e"
city = "Miami-Fort Lauderdale-Miami Beach"
country_code = "US"
start_date_1= "2022-01-01T00:00:00Z"  
end_date_1 = "2022-12-31T23:59:59Z"
# start_date_2= "2023-01-01T00:00:00Z"  
# end_date_2 = "2023-12-31T23:59:59Z"


def get_location_id(city, country_code):
    url = "https://api.openaq.org/v2/locations"
    
    # Use the correct header for OpenAQ API key
    headers = {
        "x-api-key": api_key  # Correct header for OpenAQ API
    }
    
    params = {
        "city": city,
        "country": country_code
    }

    response = requests.get(url, headers=headers, params=params)
    
    # Print response for debugging
    #print(response.status_code, response.json())

    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            return data['results'][0]['id']  # Return the first location ID
        else:
            print(f"No location found for {city}, {country_code}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def detect_anomalies_in_data(flattened_data):
    anomalies = []
    for entry in flattened_data:
        # Check for missing critical values
        if entry['date'] is None or entry['parameter'] is None or entry['value'] is None:
            anomalies.append(f"Missing data detected in entry: {entry}")
            continue
        # Check for invalid value ranges (e.g., negative pollution values)
        if entry['parameter'] in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'] and entry['value'] < 0:
            anomalies.append(f"Invalid negative value detected for {entry['parameter']}: {entry['value']} at {entry['date']}")
        
        # Check for duplicate records based on (date, location, parameter)
        if flattened_data.count(entry) > 1:
            anomalies.append(f"Duplicate entry detected: {entry}")
    return anomalies  # Return list of anomalies (empty if none detected)

def get_air_pollution_data(location_id, start_date, end_date):
    url = "https://api.openaq.org/v2/measurements"
    headers = {
        "x-api-key": api_key
    }
    params = {
        "location_id": location_id,
        "date_from": start_date,  # Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        "date_to": end_date,      # End date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        "limit": 40000,           # Adjust limit as necessary
        "sort": "asc"            # Sort by the latest data first
    }
    response = requests.get(url, headers=headers,params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def flatten_data(data):
    flattened_data = []
    for entry in data['results']:
            # Ensure that entry is a dictionary before trying to access its fields
            if isinstance(entry, dict):
                flat_entry = {
                    "date": entry.get("date", {}).get("utc", None),
                    "location": entry.get("location", None),
                    "parameter": entry.get("parameter", None),
                    "value": entry.get("value", None),
                    "unit": entry.get("unit", None),
                    "latitude": entry.get("coordinates", {}).get("latitude", None),
                    "longitude": entry.get("coordinates", {}).get("longitude", None),
                    "city": entry.get("city", None),
                    "country": entry.get("country", None)
                }
                flattened_data.append(flat_entry)
    return flattened_data

def save_to_csv(data, filename):
    flattened_data = flatten_data(data)
    df = pd.DataFrame(flattened_data)
    df.to_csv(os.path.join(os.getcwd(),filename), index=False)
    print(f"Data saved to {filename}")

def save_to_csv_gcs(data, bucket_name, destination_blob_name):
    # Initialize the storage client
    client = storage.Client()

    # Get the GCS bucket
    bucket = client.bucket(bucket_name)

    # Create a blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Flatten data and convert to a DataFrame
    flattened_data = flatten_data(data)
    df = pd.DataFrame(flattened_data)

    # Save DataFrame to a CSV file in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload CSV to the bucket
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    print(f"Data saved to GCS bucket '{bucket_name}' as '{destination_blob_name}'")

def get_available_parameters(location_id):
    url = f"https://api.openaq.org/v2/locations/{location_id}"
    headers = {
        "x-api-key": api_key
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            parameters = data['results'][0]['parameters']
            return parameters
        else:
            print(f"No parameters found for location ID {location_id}")
            return None
    else:
        print(f"Error fetching parameters: {response.status_code}")
        return None

def file_exists_in_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.exists()

def download_data_function():
    # check if anything is already present in the gcp bucket with name air_pollution_data_1.csv, if it exists, then dont push it

    client = storage.Client()

    bucket_name = 'airquality-mlops-rg'
    destination_blob_name = 'api_data/air_pollution_data_1.csv'

    if file_exists_in_gcs(bucket_name, destination_blob_name):
        print(f"File '{destination_blob_name}' already exists in bucket '{bucket_name}'. Skipping upload.")
        return

    # Get the bucket and the blob (file)
    bucket = client.bucket(bucket_name)

    location_id = 869  # Example location ID
    all_anomalies = []

    available_parameters = get_available_parameters(location_id)
    if available_parameters:
        print("Available parameters at this location:")
        for param in available_parameters:
            print(param['parameter'])

    if location_id:
        print(f"Location ID for {city}: {location_id}")
        air_pollution_data_1 = get_air_pollution_data(location_id, start_date_1, end_date_1)

    if air_pollution_data_1:
        flattened_data_1 = flatten_data(air_pollution_data_1)
        anomalies_1 = detect_anomalies_in_data(flattened_data_1)
        if anomalies_1:
            all_anomalies.extend(anomalies_1)
        # Save directly to GCS
        save_to_csv_gcs(air_pollution_data_1, bucket_name, destination_blob_name)

    return all_anomalies

# def download_data_function():
#     client = storage.Client()

#     bucket_name = 'airquality-mlops-rg'
#     csv_file_path = 'api_data/air_pollution_data_1.csv'

#     # Get the bucket and the blob (file)
#     bucket = client.bucket(bucket_name)
#     # check if anything exists in the csv bucket
#     #if not then extract for year 2022
#     # if exists then dont do anything
#     #location_id = get_location_id(city, country_code)
#     location_id = 869
#     all_anomalies = []
#     #155
#     #869
#     #1341

#     available_parameters = get_available_parameters(location_id)
#     if available_parameters:
#         print("Available parameters at this location:")
#         for param in available_parameters:
#             print(param['parameter'])

#     if location_id:
#         print(f"Location ID for {city}: {location_id}")
#         air_pollution_data_1 = get_air_pollution_data(location_id, start_date_1, end_date_1)
#         #air_pollution_data_2 = get_air_pollution_data(location_id, start_date_2, end_date_2)
        
#     if air_pollution_data_1:
#         flattened_data_1 = flatten_data(air_pollution_data_1)
#         anomalies_1 = detect_anomalies_in_data(flattened_data_1)
#         if anomalies_1:
#             all_anomalies.extend(anomalies_1)
#         save_to_csv(air_pollution_data_1, "dags/DataPreprocessing/src/data_store_pkl_files/csv/air_pollution_data_1.csv")
#     # if air_pollution_data_2:
#     #     flattened_data_2 = flatten_data(air_pollution_data_2)
#     #     anomalies_2 = detect_anomalies_in_data(flattened_data_2)
#     #     if anomalies_2:
#     #         all_anomalies.extend(anomalies_2)
#     #     save_to_csv(air_pollution_data_2, "dags/DataPreprocessing/src/data_store_pkl_files/csv/air_pollution_data_2.csv")
#     return all_anomalies

if __name__ == "__main__":
    download_data_function()