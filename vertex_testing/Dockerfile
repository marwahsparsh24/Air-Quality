# Use Python 3.8 as the base image
FROM python:3.8

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
COPY train.py /app/
COPY key.json /app/key.json
RUN pip install --no-cache-dir -r requirements.txt
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json

# Debugging: List files in the working directory during build

# Use absolute path to run the training script
ENTRYPOINT ["python", "/app/train.py"]
