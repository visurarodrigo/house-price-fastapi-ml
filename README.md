
# House Price FastAPI ML

[![CI/CD Pipeline](https://github.com/visurarodrigo/house-price-fastapi-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/visurarodrigo/house-price-fastapi-ml/actions/workflows/ci.yml)

## Overview
House Price FastAPI ML is an end-to-end tabular regression portfolio project that trains a house price prediction model and serves inference through a FastAPI application.

The project emphasizes practical ML engineering patterns:
- Modular preprocessing and model code
- A single serialized sklearn pipeline for inference
- Clean API contracts and test coverage
- **Docker containerization for easy deployment**
- **Automated CI/CD pipeline for code quality checks**

## Business Problem
Real-estate stakeholders often need quick, consistent property value estimates based on structured listing attributes.

This project demonstrates a baseline approach to predicting `price_lkr` from listing features, then exposing that prediction via a lightweight REST API suitable for local deployment and extension.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- pytest
- **Docker**

## Data Source
- Kaggle dataset: [House Prices in Sri Lanka](https://www.kaggle.com/datasets/dewminimnaadi/house-prices-in-sri-lanka)
- Local dataset file used in this project: `data/housing_lk.csv`

### About Dataset
Sri Lanka Synthetic Real Estate Dataset (20,000 Listings)

This dataset contains 20,000 synthetically generated real estate listings representing properties across districts in Sri Lanka. It is designed to provide a realistic but fully artificial dataset for practicing data analysis, visualization, and machine learning techniques, particularly in the context of house price prediction and property market analytics.

## Project Structure
```text
house-price-fastapi-ml/
|
|-- api/
|   |-- main.py
|   `-- Basic app.jpg
|-- data/
|   `-- housing_lk.csv
|-- models/
|   |-- .gitkeep
|   `-- house_price_pipeline.joblib
|-- notebooks/
|   |-- .gitkeep
|   |-- housing_eda.ipynb
|   `-- EDA Outputs/
|       |-- Correlation Heatmap - Numerical Features.png
|       `-- Relationship between selected numerical features and target.png
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- model.py
|   |-- train.py
|   `-- predict.py
|-- tests/
|   |-- __init__.py
|   `-- test_api.py
|-- .github/workflows/ci.yml
|-- .gitignore
|-- LICENSE
|-- requirements.txt
`-- README.md
```

## Installation
1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

## 🐳 Docker Setup
This project is fully containerized. You can run the entire application without installing Python or libraries on your machine.

### Build and Run
```bash
# Build the Docker image
docker build -t house-price-api .

# Run the container
docker run -p 8000:8000 house-price-api
```

### Access
- **API Documentation:** http://localhost:8000/docs
- **Web UI:** http://localhost:8000/app

## Training The Model
Dataset and target:
- Dataset file: `data/housing_lk.csv`
- Target column: `price_lkr`

Run training:

```bash
python -m src.train
```

Training outputs:
- Trained artifact: `models/house_price_pipeline.joblib`
- Console metrics:
	- 5-fold cross-validation on training data (`X_train`, `y_train`) using R²
	- Mean and standard deviation of CV R² scores
	- Final MAE, RMSE, and R² on the held-out test set

## Running The API (Local)
If not using Docker, start the FastAPI app:

```bash
uvicorn api.main:app --reload
```

## Basic Web Interface
For quick manual testing, open the built-in UI:

- `http://127.0.0.1:8000/app`

This page lets you enter feature values and get a predicted price without using Swagger or curl.

### App Preview
![Basic App Preview](api/Basic%20app.jpg)

## Running Tests
Run the test suite:

```bash
pytest
```

## Exploratory Data Analysis (EDA)
Notebook:
- `notebooks/housing_eda.ipynb`

Saved visual outputs:
- `notebooks/EDA Outputs/Correlation Heatmap - Numerical Features.png`
- `notebooks/EDA Outputs/Relationship between selected numerical features and target.png`

### Correlation Heatmap - Numerical Features
![Correlation Heatmap - Numerical Features](notebooks/EDA%20Outputs/Correlation%20Heatmap%20-%20Numerical%20Features.png)

### Relationship Between Selected Numerical Features And Target
![Relationship between selected numerical features and target](notebooks/EDA%20Outputs/Relationship%20between%20selected%20numerical%20features%20and%20target.png)

## API Endpoints
- `GET /`
  - Health check endpoint that confirms the service is running.
- `GET /app`
	- Minimal browser interface for manual predictions.
- `POST /predict`
  - Accepts house features and returns predicted house price in LKR.

## Example Prediction Request
```json
{
	"district": "Colombo",
	"area": "Colombo 05",
	"perch": 20,
	"bedrooms": 4,
	"bathrooms": 3,
	"kitchen_area_sqft": 180,
	"parking_spots": 2,
	"has_garden": true,
	"has_ac": true,
	"water_supply": "Pipe-borne",
	"electricity": "Three phase",
	"floors": 2,
	"year_built": 2015
}
```

## Example Prediction Response
```json
{
	"predicted_price_lkr": 61839550.97
}
```

## Model Performance
Evaluation is reported in two stages during training:
- 5-fold cross-validation on the training split (`X_train`, `y_train`) with R² scoring
- Final evaluation on the held-out test split (`X_test`, `y_test`)

Baseline model performance on held-out test data:
- MAE: about 1.77M LKR
- RMSE: about 2.72M LKR
- R2: 0.9583

These metrics indicate strong baseline explanatory performance for this dataset, while still leaving room for robustness and calibration improvements.

## Key Skills Demonstrated
- Building modular ML components for preprocessing, training, and inference
- Constructing and serializing a full sklearn pipeline
- Serving model inference with FastAPI and Pydantic schemas
- Implementing endpoint-level API tests with pytest and TestClient
- **Containerizing ML applications with Docker**
- **Setting up CI/CD pipelines with GitHub Actions**

## Future Improvements
- Add experiment tracking and model versioning (e.g., MLflow)
- Add hyperparameter tuning and model comparison
- Add data validation and drift monitoring checks
- Add automated model retraining pipeline triggered by new data
- Deploy to a cloud platform (AWS/GCP) with monitoring

Author - Visura Rodrigo