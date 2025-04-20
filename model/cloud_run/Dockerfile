# Use a lightweight Python base image
FROM python:3.8-slim

ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
ENV MLFLOW_TRACKING_DIR=/app/mlruns

# Set the working directory inside the container
WORKDIR /app

# Install dependencies for Google Cloud SDK and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc-dev jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ${MLFLOW_TRACKING_DIR}

# Copy application files
COPY random_forest_train.py /app/
COPY Prophet_train.py /app/
COPY XGBoost_train.py /app/
COPY RandomForest_Valid.py /app/
COPY Prophet_Valid.py /app/
COPY XGBoost_valid.py /app/
COPY Model_bias.py /app/
COPY bestmodel.py /app/
COPY saving_bigquery.py /app/
COPY delete_table.py /app/
COPY testing.py /app/


RUN chmod -R 777 ${MLFLOW_TRACKING_DIR}

# ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service_account_key.json"

# Fetch the key and run the training script
#CMD ["sh", "-c", "python /app/XGBoost_train.py && python /app/XGBoost_valid.py && python /app/random_forest_train.py && python /app/RandomForest_Valid.py && python /app/Model_bias.py && python /app/bestmodel.py"]
# CMD ["sh", "-c", "python /app/Prophet_train.py && python /app/Prophet_Valid.py && python /app/XGBoost_train.py && python /app/XGBoost_valid.py && python /app/random_forest_train.py && python /app/RandomForest_Valid.py && python /app/Model_bias.py && python /app/bestmodel.py"]
CMD ["sh", "-c", "python /app/delete_table.py && python /app/saving_bigquery.py && python /app/Prophet_train.py && python /app/Prophet_Valid.py && python /app/XGBoost_train.py && python /app/XGBoost_valid.py && python /app/random_forest_train.py && python /app/RandomForest_Valid.py && python /app/Model_bias.py && python /app/bestmodel.py"]
# CMD ["sh", "-c", "python /app/testing.py"]