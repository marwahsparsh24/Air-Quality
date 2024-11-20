# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the default entry point to execute the training script
ENTRYPOINT ["python", "train.py"]
