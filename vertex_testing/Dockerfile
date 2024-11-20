# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and any other files needed
COPY train.py /app/

# Set the default entry point to execute the training script
ENTRYPOINT ["python", "train.py"]
