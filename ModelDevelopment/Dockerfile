# Use an official Python image as the base
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY . .

# Set the MLflow tracking URI to a folder inside the container
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"

ENV PYTHONPATH="/app/ModelDevelopment:${PYTHONPATH}"

RUN mkdir -p /app/weights

RUN mkdir -p /app/artifacts

# Add execute permissions for all Python scripts in Training and Validation folders
RUN chmod +x Training/*.py Validation/*.py ModelBias/*.py

# Run scripts in sequence with 'python' in front of each
CMD ["sh", "-c", "python Training/Prophet.py && python Validation/Prophet.py && python Training/RandomForest.py && python Validation/RandomForest.py && python Training/XGBoost.py && python Validation/XGBoost.py && python ModelBias/Model_bias.py && python bestmodel.py"]
