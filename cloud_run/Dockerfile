# Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the script and requirements file
COPY train.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for GCP credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# Command to run the script
CMD ["python", "app/train.py"]
