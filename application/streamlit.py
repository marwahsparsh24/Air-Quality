import streamlit as st
import requests
import json
import math
from datetime import datetime,date,timedelta
from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery client
# change it according to the table
client = bigquery.Client(project="airquality-438719")

# Define the endpoint URL
endpoint = "https://us-central1-airquality-438719.cloudfunctions.net/predict-function/predict"

# Helper function to compute cyclic features
def compute_cyclic_features(value, max_value):
    sin_val = math.sin(2 * math.pi * value / max_value)
    cos_val = math.cos(2 * math.pi * value / max_value)
    return sin_val, cos_val

# Store data in BigQuery
def store_in_bigquery(input_data, predicted_value, predictions_table, datetime_obj):
    predictions_schema = [
        bigquery.SchemaField("input_data", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("predicted_value", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]

    try:
        client.get_table(predictions_table)
    except:
        table = bigquery.Table(predictions_table, schema=predictions_schema)
        client.create_table(table)

    rows_to_insert = [
        {
            "input_data": json.dumps(input_data),
            "predicted_value": predicted_value,
            "timestamp": datetime_obj.isoformat(),
        }
    ]
    errors = client.insert_rows_json(predictions_table, rows_to_insert)
    if errors:
        st.error(f"Failed to store data in BigQuery: {errors}")

def find_date_in_bigquery(table_id, datetime_obj, max_attempts=10):
    for attempt in range(max_attempts):
        datetime_iso = datetime_obj.isoformat()
        print(f"Checking date: {datetime_iso}")
        query = f"""
        SELECT COUNT(*) as date_count
        FROM `{table_id}`
        WHERE timestamp = '{datetime_iso}'
        """
        query_job = client.query(query)
        results = query_job.result()
        date_count = list(results)[0].date_count

        if date_count > 0:
            print(f"Date {datetime_iso} found in the table.")
            return datetime_iso  # Return the found date

        print(f"Date {datetime_iso} not found. Decrementing year...")
        datetime_obj = datetime_obj.replace(year=datetime_obj.year - 1)

    print(f"No matching date found after {max_attempts} attempts.")
    return None

def get_feature_data_for_date(table_id, datetime_iso):
    query = f"""
    SELECT feature_data
    FROM `{table_id}`
    WHERE timestamp = '{datetime_iso}'
    """
    query_job = client.query(query)
    results = query_job.result()

    rows = list(results)
    if rows:
        feature_data = rows[0].feature_data  # Assuming `feature_data` column exists
        return feature_data
    return None

def main():
    min_date = date(2022, 1, 1)
    st.title("Air Quality Prediction")
    st.header("Enter Date and Time for Prediction")
    input_date = st.date_input("Select a date for prediction:",min_value=min_date)
    input_time = st.time_input("Select a time for prediction:")
    additional_days = st.slider(
        "Select number of days for additional predictions (1-10):",
        min_value=0,
        max_value=10,
        value=0,
    )
    plot_placeholder = st.empty()
    result_placeholder = st.empty()
    # if no value is selected just make the prediction and say if it is good or bad, something artistic using gpt key
    # if value is selected build a plot and keep it in the place holder and describe something aritistic using gpt key, explain plot
    # everytime predict is pressed remove everything in the place holders
    if st.button("Predict Air Quality"):
        plot_placeholder.empty()  # Clears any existing plot
        result_placeholder.empty()
        try:
            datetime_obj = datetime.combine(input_date, input_time)
            table_id = "airquality-438719.airqualityuser.predictions"
            predictions = []
            for i in range(additional_days + 1):
                current_datetime = datetime_obj + timedelta(days=i)
                found_date = find_date_in_bigquery(table_id, current_datetime)
                feature_data = get_feature_data_for_date(table_id, found_date)
                day_of_week = datetime_obj.weekday()
                day_of_year = datetime_obj.timetuple().tm_yday
                month = datetime_obj.month
                hour = datetime_obj.hour
                sin_hour, cos_hour = compute_cyclic_features(hour, 24)
                sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)
            
                payload = {
                    "instances": [
                        {
                            "lag_1": feature_data["lag_1"],
                            "lag_2": feature_data["lag_2"],
                            "lag_3": feature_data["lag_3"],
                            "lag_4": feature_data["lag_4"],
                            "lag_5": feature_data["lag_5"],
                            "rolling_mean_3": feature_data["rolling_mean_3"],
                            "rolling_mean_6": feature_data["rolling_mean_6"],
                            "rolling_mean_24": feature_data["rolling_mean_24"],
                            "rolling_std_3": feature_data["rolling_std_3"],
                            "rolling_std_6": feature_data["rolling_std_6"],
                            "rolling_std_24": feature_data["rolling_std_24"],
                            "ema_3": feature_data["ema_3"],
                            "ema_6": feature_data["ema_6"],
                            "ema_24": feature_data["ema_24"],
                            "diff_1": feature_data["diff_1"],
                            "diff_2": feature_data["diff_2"],
                            "hour": hour,
                            "day_of_week": day_of_week,
                            "day_of_year": day_of_year,
                            "month": month,
                            "sin_hour": sin_hour,
                            "cos_hour": cos_hour,
                            "sin_day_of_week": sin_day_of_week,
                            "cos_day_of_week": cos_day_of_week,
                        }
                    ]
                }

                headers = {"Content-Type": "application/json"}
                response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

                if response.status_code == 200:
                    prediction = response.json()
                    predicted_value = prediction["predictions"][0]
                    predictions.append({"date": current_datetime, "value": predicted_value})
                    st.success("Prediction Successful!")
                    st.write(f"Predicted Air Quality Value: {predicted_value}")
                    predictions_table = "airquality-438719.airqualityuser.predictions"
                    store_in_bigquery(payload["instances"][0], predicted_value, predictions_table, datetime_obj)
                else:
                    result_placeholder.error(f"Error {response.status_code}: {response.text}")

            if additional_days == 0:
                predicted_value = predictions[0]["value"]
                result_placeholder.success(f"Prediction Successful! Air Quality: {predicted_value}")
                artistic_description = f"The air quality on {datetime_obj.date()} is expected to be {'Good' if predicted_value < 5 else 'Bad'}."
                result_placeholder.write(artistic_description)
            else:
                # Plot results
                with plot_placeholder:
                    df = pd.DataFrame(predictions)
                    st.line_chart(data=df, x="date", y="value")
                    artistic_description = f"The air quality predictions for the next {additional_days} days show these trends."
                    result_placeholder.write(artistic_description)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

