# Models Directory

This folder is used to store trained machine learning model artifacts.

## Why is the model file ignored in Git?

The trained model file (`house_price_pipeline.joblib`) is intentionally excluded from version control for the following reasons:

- **Large file size**: Model files can be large and may slow down the repository.
- **Reproducibility**: The model can be regenerated anytime using the training script (`src/train.py`).
- **Avoid unnecessary updates**: Model files may change frequently and are not source code.
- **Best practice**: Version control should track code, not generated artifacts.

## How to regenerate the model

Run the training script:

```bash
python -m src.train