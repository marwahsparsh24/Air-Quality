from google.cloud import bigquery
from datetime import datetime

# Initialize BigQuery client
client = bigquery.Client(project="airquality-438719")

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

table_id = "airquality-438719.airqualityuser.allfeatures"
datetime_obj = datetime(2025, 1, 6, 11, 0, 0)
found_date = find_date_in_bigquery(table_id, datetime_obj)
if found_date:
    feature_data = get_feature_data_for_date(table_id, found_date)
    print(feature_data)
    if feature_data:
        print(f"Feature data for {found_date}: {feature_data}")
    else:
        print(f"No feature data found for {found_date}.")
else:
    print("No matching date found.")

