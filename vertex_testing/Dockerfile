# Use a lightweight Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc-dev jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies first (caching layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (train.py, etc.)
COPY train.py /app/
COPY fetch_key_json.py /app/  

# Add support for Google credentials
ARG GOOGLE_APPLICATION_CREDENTIALS
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service_account_key.json

# Entry point to validate and fetch the service account key before running the training script
ENTRYPOINT ["sh", "-c", "if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]; then echo 'Credentials validated'; else echo 'Missing credentials at $GOOGLE_APPLICATION_CREDENTIALS' && exit 1; fi && python /app/train.py"]
