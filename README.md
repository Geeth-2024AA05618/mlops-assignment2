### Cats vs Dogs Classification – End-to-End MLOps Pipeline
## Project Overview

This project implements a complete MLOps lifecycle for a Cats vs Dogs image classification model.

It includes:

- Data versioning with DVC
- Model training with MLflow tracking
- Containerization using Docker
- CI/CD pipeline using GitHub Actions
- Automated testing using Pytest
- Deployment with Docker
- Monitoring and performance validation

The objective is to demonstrate a production-ready ML workflow from data ingestion to deployment and monitoring.

## Project Architecture

Raw Data → Preprocessing → Model Training → MLflow Tracking
      ↓
    DVC Versioning
      ↓
Docker Container → API (FastAPI)
      ↓
CI/CD Pipeline (GitHub Actions)
      ↓
Deployment + Health Check
      ↓
Monitoring & Performance Tracking

## Project structure
├── api/                  # FastAPI inference service
├── src/                  # Training & preprocessing code
├── monitoring/           # Performance & monitoring scripts
├── tests/                # Unit tests
├── data/                 # DVC tracked data
├── models/               # Trained model
├── .github/workflows/    # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── README.md

## Data versioning (DVC)
Raw dataset stored in data/raw
Preprocessed dataset stored in data/processed
DVC used to track large data files
Remote storage configured
Data can be restored using


## ML Model and ML Flow building 
Features:
CNN model built with TensorFlow/Keras
MLflow experiment tracking enabled
Metrics logged (accuracy, loss)
Model artifact saved as models/model.keras

Run training: python src/train.py

## Launch MLflow UI:
mlflow ui

Access at: http://localhost:5000

## Automated Testing

Unit tests implemented using Pytest:
- Data preprocessing function test
- Inference utility test

Run tests: pytest

## M3 – Docker Containerization

The model inference service is containerized.

Build image: docker build -t cats-dogs-api .

Run container: docker run -p 8000:8000 cats-dogs-api

Access API: http://localhost:8000/health

## M4 – CI/CD Pipeline

Implemented using GitHub Actions.

Pipeline stages:
- Install dependencies
- Run tests
- Build Docker image
- Push image to GitHub Container Registry (GHCR)
- Deploy container
- Health check validation

Pipeline triggers on:
- Push to main branch
- Deployment automatically updates the running service.

## M5 – Monitoring & Logging
- Logs prediction label
- Logs request latency
- Tracks total requests

Metrics Endpoint - GET /metrics

Returns:
{
  "total_requests": 10,
  "average_latency": 0.134
}
 ## Post-Deployment Performance Check

Script: monitoring/performance_check.py
- Sends sample images to deployed API
- Compares predictions with true labels
- Calculates post-deployment accuracy

Run: python monitoring/performance_check.py

## API Endpoints
Health Check
GET /health
Response: {"status": "Model is running"}

Prediction
POST /predict
Input: Image file
Response:
{
  "prediction": "Dog",
  "confidence": 0.87,
  "latency_seconds": 0.142
}

Metrics
GET /metrics
Returns request statistics.

## Tech Stack

Python 3.12

TensorFlow / Keras

FastAPI

DVC

MLflow

Docker

GitHub Actions

Pytest


## To Run in local
1. Setup Env
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Pull data - dvc pull

3. Train model - python src/train.py

4. Start API - uvicorn api.app:app --reload

5. Test API - http://localhost:8000/health
