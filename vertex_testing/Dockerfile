# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
COPY train.py /app/
RUN pip install --no-cache-dir -r requirements.txt

# Debugging: List files in the working directory during build
RUN ls -l /app

# Use absolute path to run the training script
ENTRYPOINT ["python", "/app/train.py"]
