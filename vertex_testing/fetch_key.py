from google.cloud import secretmanager
import os
import json

def fetch_key_json(secret_name, output_path="/app/key.json"):
    """
    Fetches a secret from Google Secret Manager and writes it as a JSON file.

    Args:
        secret_name (str): The name of the secret in Google Secret Manager.
        output_path (str): Path to save the secret as a file (default: "/app/key.json").
    """
    client = secretmanager.SecretManagerServiceClient()
    
    # Retrieve the project ID from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not project_id:
        raise ValueError("Project ID is not set in environment variables.")
    
    # Construct the secret path
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    
    # Access the secret from Secret Manager
    try:
        response = client.access_secret_version(name=secret_path)
        secret_data = response.payload.data.decode("UTF-8")
        
        # Validate and write the secret to a file
        try:
            json_data = json.loads(secret_data)  # Validate JSON format
            with open(output_path, "w") as key_file:
                json.dump(json_data, key_file, indent=2)  # Write formatted JSON
            print(f"Secret successfully written to {output_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Retrieved secret is not valid JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Error accessing secret version: {e}")

if __name__ == "__main__":
    SECRET_NAME = "key"  # Replace with your secret name
    OUTPUT_PATH = "/app/service_account_key.json"

    fetch_key_json(secret_name=SECRET_NAME, output_path=OUTPUT_PATH)
