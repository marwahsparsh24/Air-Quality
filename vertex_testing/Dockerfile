# Use a lightweight Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies first (caching layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (train.py, etc.)
COPY train.py /app/

# Build-time ARG for GCP Service Account Key
ARG GCP_SA_KEY
RUN echo "$GCP_SA_KEY" > /app/key.json

# Environment variable for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"

# Run the training script when the container starts
ENTRYPOINT ["python", "/app/train.py"]
