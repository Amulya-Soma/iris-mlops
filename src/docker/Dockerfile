# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire src folder into /app
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Environment variables for MLflow tracking (update URI if needed)
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:8000
ENV MLFLOW_EXPERIMENT_NAME=iris_classification

# Run FastAPI app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

