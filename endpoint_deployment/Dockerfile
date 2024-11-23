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
COPY flask.py /app/

# Expose the application port
EXPOSE 8080

# Use Python to run the app (or replace with Gunicorn for production)
CMD ["python", "flask.py"]
