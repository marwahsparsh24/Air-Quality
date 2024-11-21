from google.cloud import secretmanager
import os

def fetch_key_json(secret_name, output_path="/app/service_account_key.json"):
    """
    Fetch a service account key from Secret Manager and write it to a file.

    Args:
        secret_name (str): The name of the secret in Secret Manager.
        output_path (str): The path to save the fetched key.
    """
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GCP_PROJECT")
    if not project_id:
        raise ValueError("Environment variable 'GCP_PROJECT' is not set.")

    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=secret_path)

    # Decode and save the key
    secret_data = response.payload.data.decode("UTF-8")
    with open(output_path, "w") as key_file:
        key_file.write(secret_data)
    print(f"Service account key saved to {output_path}")


if __name__ == "__main__":
    # Fetch key using secret name provided as an environment variable
    SECRET_NAME = "key"
    secret_name = os.getenv("SECRET_NAME", "service-account-key")
    fetch_key_json(secret_name)


 