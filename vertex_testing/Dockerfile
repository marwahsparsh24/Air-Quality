# Use a lightweight Python base image
FROM python:3.8-slim

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

# Copy application files
COPY train.py /app/
COPY fetch_key.py /app/

# Environment variables for secret fetching
ENV GCP_PROJECT="airquality-438719"
ENV SECRET_NAME="key"
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service_account_key.json"

# Fetch the key and run the training script
ENTRYPOINT ["sh", "-c", "python /app/train.py"]

