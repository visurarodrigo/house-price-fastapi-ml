"""FastAPI application entry point for serving model predictions.

Run locally with:
	uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/app", response_class=HTMLResponse)
def prediction_ui() -> str:
	"""Serve a minimal web interface for local manual prediction testing."""
	return """
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<title>House Price Predictor</title>
	<style>
		body { font-family: Segoe UI, Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }
		h1 { margin-bottom: 4px; }
		p { margin-top: 0; color: #444; }
		.helper { margin-top: -4px; margin-bottom: 8px; font-size: 13px; color: #666; }
		.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
		label { display: block; margin-bottom: 4px; font-size: 14px; }
		input, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
		button { margin-top: 14px; padding: 10px 14px; border: none; border-radius: 6px; cursor: pointer; background: #0f62fe; color: #fff; }
		.result { margin-top: 16px; font-size: 18px; font-weight: 600; }
		@media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
	</style>
</head>
<body>
	<h1>House Price Predictor</h1>
	<p>Enter home details to estimate the property price in LKR.</p>
	<p class="helper">Tip: Keep values realistic for your location to get better estimates.</p>

	<form id="predict-form" class="grid">
		<div><label>District</label><input name="district" value="Colombo" placeholder="e.g., Colombo" required /></div>
		<div><label>Area / Neighborhood</label><input name="area" value="Colombo 05" placeholder="e.g., Colombo 05" required /></div>
		<div><label>Land Size (Perches)</label><input name="perch" type="number" value="20" min="1" required /></div>
		<div><label>Number of Bedrooms</label><input name="bedrooms" type="number" value="4" min="1" required /></div>
		<div><label>Number of Bathrooms</label><input name="bathrooms" type="number" value="3" min="1" required /></div>
		<div><label>Kitchen Size (sqft)</label><input name="kitchen_area_sqft" type="number" value="180" min="1" required /></div>
		<div><label>Parking Spaces</label><input name="parking_spots" type="number" value="2" min="0" required /></div>
		<div><label>Garden Available?</label><select name="has_garden"><option value="true">Yes</option><option value="false">No</option></select></div>
		<div><label>Air Conditioning Available?</label><select name="has_ac"><option value="true">Yes</option><option value="false">No</option></select></div>
		<div><label>Water Supply Type</label><input name="water_supply" value="Pipe-borne" placeholder="e.g., Pipe-borne" required /></div>
		<div><label>Electricity Type</label><input name="electricity" value="Three phase" placeholder="e.g., Three phase" required /></div>
		<div><label>Number of Floors</label><input name="floors" type="number" value="2" min="1" required /></div>
		<div><label>Year Built</label><input name="year_built" type="number" value="2015" min="1900" max="2100" required /></div>

		<div style="grid-column: 1 / -1;">
			<button type="submit">Predict Price</button>
			<div id="result" class="result"></div>
		</div>
	</form>

	<script>
		const form = document.getElementById("predict-form");
		const result = document.getElementById("result");

		const toBool = (value) => String(value).toLowerCase() === "true";

		form.addEventListener("submit", async (event) => {
			event.preventDefault();

			const formData = new FormData(form);
			const payload = {
				district: String(formData.get("district")),
				area: String(formData.get("area")),
				perch: Number(formData.get("perch")),
				bedrooms: Number(formData.get("bedrooms")),
				bathrooms: Number(formData.get("bathrooms")),
				kitchen_area_sqft: Number(formData.get("kitchen_area_sqft")),
				parking_spots: Number(formData.get("parking_spots")),
				has_garden: toBool(formData.get("has_garden")),
				has_ac: toBool(formData.get("has_ac")),
				water_supply: String(formData.get("water_supply")),
				electricity: String(formData.get("electricity")),
				floors: Number(formData.get("floors")),
				year_built: Number(formData.get("year_built"))
			};

			result.textContent = "Predicting...";

			try {
				const response = await fetch("/predict", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(payload)
				});

				const data = await response.json();

				if (!response.ok) {
					result.textContent = "Error: " + (data.detail || "Prediction failed");
					return;
				}

				result.textContent =
					"Estimated price: LKR " + Number(data.predicted_price_lkr).toLocaleString(undefined, { maximumFractionDigits: 2 });
			} catch (error) {
				result.textContent = "Error: unable to reach API";
			}
		});
	</script>
</body>
</html>
"""


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
