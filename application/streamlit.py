import streamlit as st
import requests
import json
import math
from datetime import datetime,date,timedelta,time
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
    return datetime_iso

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
    
    # day_of_week = datetime_iso.weekday()
    # day_of_year = datetime_iso.timetuple().tm_yday
    # month = datetime_iso.month
    # hour = datetime_iso.hour
    # sin_hour, cos_hour = compute_cyclic_features(hour, 24)
    # sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)
    feature = {
                "lag_1": 2.1462424282737667,
                "lag_2": 2.1260022701562704,
                "lag_3": 1.9535269514369231,
                "lag_4": 1.90712065179672,
                "lag_5": 1.90712065179672,
                "rolling_mean_3": 2.090259560846853,
                "rolling_mean_6": 2.006424489595154,
                "rolling_mean_24": 2.0784056585593538,
                "rolling_std_3": 0.08007872420850096,
                "rolling_std_6": 0.10623635498573898,
                "rolling_std_24": 0.15142463206132026,
                "ema_3": 2.099382685440239,
                "ema_6": 2.0683034409780907,
                "ema_24": 2.0814968354677896,
                "diff_1": 0.020240158117496243,
                "diff_2": 0.14770844416324347,
                "hour": 0,
                "day_of_week": 0,
                "day_of_year": 0,
                "month": 0,
                "sin_hour": 0,
                "cos_hour": 0,
                "sin_day_of_week": 0,
                "cos_day_of_week": 0,
                }
    return feature

def generate_time_options():
    """Generate a static list of time options with 1-hour intervals."""
    return [time(hour=h, minute=0) for h in range(24)]  # Times: 00:00, 01:00, ..., 23:00

def main():
    min_date = date(2022, 1, 1)
    st.title("Air Quality Prediction")
    st.header("Enter Date and Time for Prediction")
    time_options = generate_time_options()
    input_date = st.date_input("Select a date for prediction:",min_value=min_date)
    input_time = st.selectbox("Select a time for prediction:", options=time_options)
    additional_days = st.slider(
        "Select number of hours for additional predictions (1-24):",
        min_value=0,
        max_value=23,
        value=0,
    )
    plot_placeholder = st.empty()
    result_placeholder = st.empty()
    if st.button("Predict Air Quality"):
        plot_placeholder.empty()
        result_placeholder.empty()
        datetime_obj = datetime.combine(input_date, input_time)
        table_id = "airquality-438719.airqualityuser.allfeatures"
        predictions = []
        for i in range(additional_days + 1):
            current_datetime = datetime_obj + timedelta(hours=i)
            found_date = find_date_in_bigquery(table_id, current_datetime)
            feature_data_entry = get_feature_data_for_date(table_id, found_date)
            day_of_week = current_datetime.weekday()
            day_of_year = current_datetime.timetuple().tm_yday
            month = current_datetime.month
            hour = current_datetime.hour
            sin_hour, cos_hour = compute_cyclic_features(hour, 24)
            sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)
            if isinstance(feature_data_entry, str):
                feature_data_dict = json.loads(feature_data_entry)
            elif isinstance(feature_data_entry, dict):
                feature_data_dict = feature_data_entry
            # feature_data_dict = feature_data_entry
            # feature_data_dict  = json.loads(feature_data_entry)
            payload = {
                "instances": [
                    {
                        "lag_1": feature_data_dict["lag_1"],
                        "lag_2": feature_data_dict["lag_2"],
                        "lag_3": feature_data_dict["lag_3"],
                        "lag_4": feature_data_dict["lag_4"],
                        "lag_5": feature_data_dict["lag_5"],
                        "rolling_mean_3": feature_data_dict["rolling_mean_3"],
                        "rolling_mean_6": feature_data_dict["rolling_mean_6"],
                        "rolling_mean_24": feature_data_dict["rolling_mean_24"],
                        "rolling_std_3": feature_data_dict["rolling_std_3"],
                        "rolling_std_6": feature_data_dict["rolling_std_6"],
                        "rolling_std_24": feature_data_dict["rolling_std_24"],
                        "ema_3": feature_data_dict["ema_3"],
                        "ema_6": feature_data_dict["ema_6"],
                        "ema_24": feature_data_dict["ema_24"],
                        "diff_1": feature_data_dict["diff_1"],
                        "diff_2": feature_data_dict["diff_2"],
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
                predictions_table = "airquality-438719.airqualityuser.predictions"
                store_in_bigquery(payload["instances"][0], predicted_value, predictions_table, datetime_obj)
            else:
                result_placeholder.error(f"Error {response.status_code}: {response.text}").error(f"Error {response.status_code}: {response.text}")

        if additional_days == 0:
            predicted_value = predictions[0]["value"]
            if predicted_value < 2:
                air_quality = "Good"
            elif 2 <= predicted_value < 5:
                air_quality = "Moderate"
            else:
                air_quality = "Bad"
            artistic_description = f"Air Quality: {predicted_value}. The air quality on {datetime_obj.date()} is expected to be '{air_quality}'."
            result_placeholder.write(artistic_description)
        else:
            # Plot results
            st.success("Prediction Successful!")
            for prediction in predictions:
                dates = prediction["date"].strftime("%Y-%m-%d %H:%M:%S")  # Format datetime
                value = prediction["value"]
                st.write(f"Time: {dates}, Predicted Value: {value}")
            with plot_placeholder:
                # analyse the graph using gpt
                df = pd.DataFrame(predictions)
                average_value = df["value"].mean()
                min_value = df["value"].min()
                max_value = df["value"].max()
                good_hours = len(df[df["value"] < 2])
                moderate_hours = len(df[(df["value"] >= 2) & (df["value"] < 5)])
                bad_hours = len(df[df["value"] >= 5])
                st.line_chart(data=df, x="date", y="value")
                artistic_description = f"""
                The air quality predictions for the next {additional_days} hours show these trends.
                Here is the summary:
                - Average air quality value: {average_value:.2f}
                - Minimum value: {min_value:.2f}
                - Maximum value: {max_value:.2f}
                - Hours with 'Good' quality (<2): {good_hours}
                - Hours with 'Moderate' quality (2-5): {moderate_hours}
                - Hours with 'Bad' quality (>=5): {bad_hours}"""
                result_placeholder.write(artistic_description)
    # except Exception as e:
        #     st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

