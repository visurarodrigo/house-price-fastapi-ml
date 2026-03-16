"""Basic API tests for the FastAPI prediction service."""

from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "district": "Colombo",
    "area": "Colombo 05",
    "perch": 20,
    "bedrooms": 4,
    "bathrooms": 3,
    "kitchen_area_sqft": 180,
    "parking_spots": 2,
    "has_garden": True,
    "has_ac": True,
    "water_supply": "Pipe-borne",
    "electricity": "Three phase",
    "floors": 2,
    "year_built": 2015,
}


def test_root_endpoint_returns_200() -> None:
    """GET / should return a healthy status and message."""
    # Lightweight smoke test to verify the API process is reachable.
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_with_valid_payload_returns_prediction() -> None:
    """POST /predict with a valid payload should return a numeric prediction."""
    # Mock model inference so this test does not depend on local artifact files.
    with patch("api.main.predict_price", return_value=61839550.97):
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_lkr" in data
    assert isinstance(data["predicted_price_lkr"], float)


def test_predict_endpoint_with_invalid_payload_returns_validation_error() -> None:
    """POST /predict with missing required fields should fail validation."""
    invalid_payload = VALID_PAYLOAD.copy()
    # Remove one required field to trigger Pydantic/FastAPI validation.
    invalid_payload.pop("district")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
