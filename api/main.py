"""FastAPI application entry point for serving model predictions.

Run locally with:
	uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_price


class HouseFeatures(BaseModel):
	"""Input schema for house price prediction requests."""

	district: str
	area: str
	perch: int
	bedrooms: int
	bathrooms: int
	kitchen_area_sqft: int
	parking_spots: int
	has_garden: bool
	has_ac: bool
	water_supply: str
	electricity: str
	floors: int
	year_built: int


app = FastAPI(
	title="House Price Prediction API",
	description="Portfolio API for serving tabular regression predictions.",
	version="1.0.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
	"""Health-check endpoint indicating that the API service is running."""
	return {"message": "House price prediction API is running."}


@app.post("/predict")
def predict_house_price(payload: HouseFeatures) -> dict[str, float]:
	"""Predict house price from request payload and return value in LKR."""
	try:
		# Convert validated Pydantic model into a plain dict expected by src.predict.
		features = payload.model_dump()
		predicted_price = predict_price(features)
		return {"predicted_price_lkr": predicted_price}
	except FileNotFoundError as exc:
		# Model artifact is missing; this is a server setup issue.
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	except ValueError as exc:
		# Bad payload content or inference-time validation issue.
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		# Fallback to avoid leaking internal details to API clients.
		raise HTTPException(
			status_code=500,
			detail="Unexpected server error during prediction.",
		) from exc
