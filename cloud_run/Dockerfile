# Use a lightweight Python base image
FROM python:3.8-slim

ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
ENV MLFLOW_TRACKING_DIR=/app/mlruns

# Set the working directory inside the container
WORKDIR /app

# Install dependencies for Google Cloud SDK and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc-dev jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ${MLFLOW_TRACKING_DIR}

# Copy application files
COPY random_forest_train.py /app/

RUN chmod -R 777 ${MLFLOW_TRACKING_DIR}

# ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service_account_key.json"

# Fetch the key and run the training script
ENTRYPOINT ["sh", "-c", "python /app/random_forest_train.py"]


# CMD ["python", "app/train.py"]




# # Python image Dockerfile
# FROM python:3.8

# # Install dependencies
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# # Copy over the model requirements and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r /requirements.txt

# # Set the working directory
# WORKDIR /app

# # Copy your model training script and other necessary files
# COPY cloud_run/ cloud_run/

# # Create a directory to save the model
# RUN mkdir -p /app/model_output

# # Entry point for running the training script
# CMD ["python", "cloud_run/training.py"]