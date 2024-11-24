# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY app.py /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
