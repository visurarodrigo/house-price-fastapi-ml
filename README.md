# house-price-fastapi-ml

## Project Overview
`house-price-fastapi-ml` is a portfolio project template for training and serving a house price prediction model with FastAPI.
The repository is intentionally lightweight so you can build the full ML workflow incrementally.

## Project Structure
```text
house-price-fastapi-ml/
|
|-- data/
|-- notebooks/
|-- src/
|   |-- __init__.py
|   |-- train.py
|   |-- preprocess.py
|   |-- model.py
|   `-- predict.py
|
|-- api/
|   `-- main.py
|
|-- models/
|-- tests/
|   `-- __init__.py
|
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup Instructions
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training
Training logic will be added in `src/train.py`.
Use this module as the entry point for:
- loading data
- preprocessing features
- fitting a model
- saving artifacts to `models/`

## Running The API
API setup will be defined in `api/main.py`.
Run the FastAPI service with:

```bash
uvicorn api.main:app --reload
```

## Future Improvements
- Add robust preprocessing and feature engineering pipeline
- Add model versioning and experiment tracking
- Add request/response schemas and validation tests
- Add CI pipeline for testing and linting
- Add Docker support for containerized deployment