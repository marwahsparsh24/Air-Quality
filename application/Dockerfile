# Use the official Python slim image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY streamlit.py /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
