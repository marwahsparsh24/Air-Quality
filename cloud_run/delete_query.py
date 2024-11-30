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


from google.cloud import bigquery

client = bigquery.Client()

def delete_entries():
    table_id = "airquality-438719.airqualityuser.allfeatures"

    # Delete all rows from the table
    query = f"DELETE FROM `{table_id}` WHERE TRUE"
    job = client.query(query)
    job.result()  # Wait for the query to complete
    print("All rows deleted from the table.")

delete_entries()


# Call the function
# delete_rows_in_chunks()
