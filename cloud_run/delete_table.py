from google.cloud import bigquery
import time
client = bigquery.Client(project="airquality-438719")
def delete_table(table_id):
    try:
        # Delete the table
        client.delete_table(table_id, not_found_ok=True)
        print(f"Table {table_id} deleted successfully.")
        time.sleep(20)
        try:
            client.get_table(table_id)
            print(f"Table {table_id} still exists! Deletion not yet reflected.")
        except Exception:
            print(f"Table {table_id} confirmed as deleted.")
    except Exception as e:
        print(f"An error occurred during table deletion: {e}")
full_table_id = "airquality-438719.airqualityuser.allfeatures"
delete_table(full_table_id)

