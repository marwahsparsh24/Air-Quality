# from google.cloud import bigquery

# # Initialize the BigQuery client
# client = bigquery.Client()

# # Step 1: Delete rows in chunks until the table is empty
# def delete_rows_in_chunks():
#     table_id = "airquality-438719.airqualityuser.allfeatures"
#     chunk_size = 1000  # Number of rows to delete per iteration
#     while True:
#         # Delete rows in chunks
#         delete_query = f"DELETE FROM `{table_id}` WHERE TRUE LIMIT {chunk_size}"
#         delete_job = client.query(delete_query)
#         delete_job.result()  # Wait for the query to complete

#         # Check remaining row count
#         check_query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
#         result = client.query(check_query).result()
#         row_count = list(result)[0].row_count
#         print(f"Remaining rows: {row_count}")

#         # Break if the table is empty
#         if row_count == 0:
#             print("Table is now empty.")
#             break

# def truncate_table():
#     query = f"TRUNCATE TABLE `airquality-438719.airqualityuser.allfeatures`"
#     client.query(query).result()  # Wait for the operation to complete
#     print("Table truncated successfully.")

# truncate_table()


# from google.cloud import bigquery

# client = bigquery.Client()

# def delete_entries():
#     table_id = "airquality-438719.airqualityuser.allfeatures"

#     # Delete all rows from the table
#     query = f"DELETE FROM `{table_id}` WHERE TRUE"
#     job = client.query(query)
#     job.result()  # Wait for the query to complete
#     print("All rows deleted from the table.")

# delete_entries()


# Call the function
# delete_rows_in_chunks()



from google.cloud import bigquery
import time

# Initialize BigQuery client
client = bigquery.Client(project="airquality-438719")

def delete_table(table_id):
    """Delete a BigQuery table and validate the deletion."""
    try:
        # Delete the table
        client.delete_table(table_id, not_found_ok=True)
        print(f"Table {table_id} deleted successfully.")
        
        # Wait for the deletion to propagate
        time.sleep(15)  # Wait 5 seconds to ensure the deletion reflects

        # Validate that the table no longer exists
        try:
            client.get_table(table_id)
            print(f"Table {table_id} still exists! Deletion not yet reflected.")
        except Exception:
            print(f"Table {table_id} confirmed as deleted.")
    except Exception as e:
        print(f"An error occurred during table deletion: {e}")

# Table to delete
full_table_id = "airquality-438719.airqualityuser.allfeatures"

# Call the function
delete_table(full_table_id)

