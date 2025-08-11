# Iris Classification MLOps Pipeline

This repository contains a complete MLOps pipeline implementation for the classic Iris flower classification problem. The project demonstrates how to take a machine learning model from development to production using modern MLOps tools and best practices.

## Features

* **Data and Code Versioning:** Organized repository with Git and optional DVC support for dataset tracking.
* **Experiment Tracking:** MLflow integration to track model training experiments, parameters, and metrics.
* **API Deployment:** FastAPI-based REST API serving the trained Iris classification model.
* **Input Validation:** Pydantic schemas ensure robust and type-safe API requests.
* **Containerization:** Dockerfile to build a portable container image for consistent deployment.
* **CI/CD Pipeline:** Automated testing, linting, Docker image build, and deployment using GitHub Actions.
* **Logging and Monitoring:** Request logging for traceability and a `/metrics` endpoint for basic monitoring.

## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/iris-mlops.git
   cd iris-mlops
   ```

2. Build and run the Docker container:

   ```bash
   docker build -t iris-api .
   docker run -p 8000:8000 -v $(pwd)/logs:/app/logs iris-api
   ```

3. Access the API documentation at `http://localhost:8000/docs`.

4. Send POST requests to `/predict` with Iris flower features to get species predictions.

## CI/CD

The project includes a GitHub Actions workflow that automates code linting, testing, Docker image building, and pushing to Docker Hub on every push to the main branch.

## Logging

Prediction requests and outputs are logged to `logs/predictions.log` for audit and debugging.
