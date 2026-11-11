# DeliveryAI: Food Delivery Time Prediction

A small end-to-end machine learning project. A Random Forest model estimates how many minutes a food delivery will take, a FastAPI backend serves the model, and a single-page web interface lets you try it in the browser.

## Features

- Predicts delivery time (in minutes) from seven order and courier details
- Scikit-learn pipeline that handles missing values and categorical encoding, so raw inputs go straight in
- FastAPI backend with typed request validation (Pydantic) and a `/predict` endpoint
- Single-file frontend (`index.html`: HTML, CSS and vanilla JavaScript) with input validation, loading states, toast error messages and a responsive layout
- Interactive API docs generated automatically at `/docs`

## Tech stack

| Layer | Tools |
|---|---|
| Model | Python, pandas, NumPy, scikit-learn (Random Forest Regressor) |
| Backend | FastAPI, Uvicorn, Pydantic, joblib |
| Frontend | HTML5, CSS3, vanilla JavaScript |

## Project structure

```
.
├── main.py                 # FastAPI app (serves the UI and the /predict endpoint)
├── model.pkl               # Trained model (saved RandomizedSearchCV object)
├── index.html              # Frontend, served at GET /
├── Random_Forest_cc.ipynb  # Training notebook
└── README.md
```

## Inputs

| Field | Type | Description |
|---|---|---|
| `Weather` | string | Weather conditions (e.g. Clear, Rainy, Foggy, Windy) |
| `Traffic_level` | string | Traffic level (e.g. Low, Medium, High) |
| `Time_of_Day` | string | Morning, Afternoon, Evening or Night |
| `Vehicle_Type` | string | Courier vehicle (e.g. Bike, Scooter) |
| `Distance_km` | float | Distance from restaurant to customer, in km |
| `Preparation_Time_min` | integer | Food preparation time, in minutes |
| `Courier_Experience_yrs` | float | Courier experience, in years |

The category lists above are examples. Use the values that appear in your training data (`df[col].unique()`).

## Machine learning workflow

The full workflow is in `Random_Forest_cc.ipynb`.

1. **Data:** `Food_Delivery_Times.csv`. Four columns (Weather, Traffic_Level, Time_of_Day, Courier_Experience_yrs) contained missing values.
2. **Target transform:** `Delivery_Time_min` was right-skewed, so the model is trained on its **natural log**. Predictions are converted back with `np.exp` in the API.
3. **Preprocessing** (`ColumnTransformer`):
   - Categorical columns: most-frequent imputation, then `OneHotEncoder(handle_unknown='ignore')`
   - Numerical columns: median imputation
4. **Baseline:** Decision Tree (`max_depth=5`), test R² ≈ 0.748
5. **Final model:** Random Forest inside a `Pipeline`, tuned with `RandomizedSearchCV` (20 iterations, 5-fold CV, scoring R², random state 42). Test R² ≈ 0.851 on the held-out 20% split, measured on the log-scale target.
6. **Saved** with `joblib.dump(rf_search, 'model.pkl')`.

## Getting started

**Requirements:** Python 3.9+

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

Use the same scikit-learn version that was used to train `model.pkl`, otherwise loading the model may fail or warn.

**Run the server** from the project folder:

```bash
uvicorn main:app --reload
```

Then open:

- App: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs

If port 8000 is already in use, start on another port with `--port 8001`.

## API reference

### `GET /`
Serves the web interface (`index.html`).

### `POST /predict`

Request body:

```json
{
  "Weather": "Rainy",
  "Traffic_level": "High",
  "Time_of_Day": "Evening",
  "Vehicle_Type": "Bike",
  "Distance_km": 7.5,
  "Preparation_Time_min": 15,
  "Courier_Experience_yrs": 3
}
```

Response:

```json
{ "delivery_time_min": 38.42 }
```

The example response is illustrative. Invalid input returns HTTP 422 with details about the offending field.

## Implementation notes

- **Log transform:** the model outputs log-minutes, so the API applies `np.exp` before returning a result.
- **Column naming:** the model was trained on `Traffic_Level`, while the API accepts `Traffic_level`. The API renames the column before calling the model.
- **Unseen categories:** the encoder ignores unknown categories. A value like `"Unknown"` is not imputed, it simply contributes no category signal.
- **Frontend API address:** set in one place, `CONFIG.API_BASE`, at the top of the script in `index.html`.

## Limitations

- Accuracy figures come from a single train/test split of one dataset, so real-world performance may differ.
- The model only knows the categories and ranges present in its training data.
- Predictions are estimates, not guarantees.

## Possible next steps

- Report error in minutes (MAE) alongside R²
- Add a `requirements.txt` with pinned versions
- Containerize with Docker
- Deploy the API and frontend

## Author

Shreyash Singh Rawat, B.Tech CSE (AI & ML)
