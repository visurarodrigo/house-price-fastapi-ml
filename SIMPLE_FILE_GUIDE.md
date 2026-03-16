# Simple File Guide (Very Easy)

This file explains what each file/folder does in very simple words.

## Big Idea of This Project
This project learns house prices from old data, saves the trained model, and gives predictions through a FastAPI API.

## What Happens (Step by Step)
1. You train the model using `src/train.py`.
2. Training reads data from `data/housing_lk.csv`.
3. It cleans/transforms data using `src/preprocess.py`.
4. It creates a model from `src/model.py`.
5. It combines preprocessing + model into one pipeline and saves it to `models/house_price_pipeline.joblib`.
6. You run the API from `api/main.py`.
7. API receives house details at `/predict`.
8. API calls `src/predict.py`, which loads the saved pipeline and predicts price.
9. API returns predicted price in LKR.

## File-by-File Explanation

### Root Files
- `README.md`:
  Main project document. Explains setup, training, API run, and examples.

- `requirements.txt`:
  List of Python packages needed for this project.

- `LICENSE`:
  Legal permission file (MIT License) for using/sharing code.

### `api/`
- `api/main.py`:
  FastAPI app entry point.
  - `GET /` = health check (service running)
  - `GET /app` = simple browser form to test predictions
  - `POST /predict` = send house features, get price prediction

- `api/__pycache__/`:
  Auto-generated Python cache files. Not part of your core logic.

### `data/`
- `data/housing_lk.csv`:
  Training dataset (input data). Contains house features + target price.

### `models/`
- `models/house_price_pipeline.joblib`:
  Saved trained pipeline file (preprocessing + model together).
  This file is used during prediction.

### `notebooks/`
- `notebooks/housing_eda.ipynb`:
  Jupyter notebook for exploratory data analysis (EDA).

- `notebooks/EDA Outputs/`:
  Saved charts/images generated from EDA notebook.

### `src/`
- `src/__init__.py`:
  Marks `src` as a Python package.

- `src/preprocess.py`:
  Data helper utilities:
  - load CSV/parquet
  - split features and target
  - detect numeric/categorical columns
  - build sklearn preprocessing pipeline

- `src/model.py`:
  Defines ML model builder (`RandomForestRegressor`).

- `src/train.py`:
  Full training flow:
  - load data
  - split train/test
  - build preprocessor + model pipeline
  - train pipeline
  - evaluate metrics (MAE, RMSE, R2)
  - save `.joblib` model artifact

- `src/predict.py`:
  Inference utilities:
  - load saved pipeline
  - convert input dict to DataFrame
  - run prediction
  - return price as float

- `src/__pycache__/`:
  Auto-generated Python cache files.

### `tests/`
- `tests/__init__.py`:
  Marks `tests` as a Python package.

- `tests/test_api.py`:
  API tests:
  - checks root endpoint
  - checks `/predict` with valid data
  - checks validation error with bad payload

- `tests/__pycache__/`:
  Auto-generated Python cache files.

## Simple Summary
- `src/train.py` creates the model file.
- `api/main.py` serves the API.
- `src/predict.py` uses saved model to predict.
- `tests/test_api.py` confirms endpoints work.
- Everything starts from data in `data/housing_lk.csv`.
