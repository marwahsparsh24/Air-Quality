from google.cloud import bigquery
from datetime import datetime
#2022-01-06 11:00:00 UTC

# Initialize BigQuery client
client = bigquery.Client(project="airquality-438719")

def query_bigquery_with_datetime(table_id, datetime_obj):
    datetime_iso = datetime_obj.isoformat()
    print(datetime_iso)
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
        return True
    else:
        print(f"Date {datetime_iso} not found in the table.")
        return False

# Example usage
table_id = "airquality-438719.airqualityuser.allfeatures"
datetime_obj = datetime(2025, 1, 6, 11, 0, 0)  # Example datetime object

# Query the table
exists = query_bigquery_with_datetime(table_id, datetime_obj)
