# Use a lightweight Python image
FROM python:3.8-slim

# Set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc-dev jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_flask.py /app/
COPY /tmp/gcp-key.json /tmp/gcp-key.json
RUN gcloud auth activate-service-account --key-file=/tmp/gcp-key.json
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk
# COPY model.pkl /app/
# COPY /tmp/gcp-key.json /app/

# Expose the application port
EXPOSE 8080

# Use Python to run the app (or replace with Gunicorn for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app_flask:app"]

# CMD ["sh", "-c", "python app_flask.py  --server.port 8080"]
