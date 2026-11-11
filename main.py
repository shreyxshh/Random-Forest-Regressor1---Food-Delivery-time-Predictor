from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import FileResponse
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class DeliveryInput(BaseModel):

    #CATEGORICAL COLUMNS 
    Weather : str = 'Unknown'
    Traffic_Level : str = 'Unknown'
    Time_of_Day : str = 'Unknown'
    Vehicle_Type : str = 'Unknown'

    #NUMERICAL COLUMNS 
    Distance_km : float
    Preparation_Time_min : int
    Courier_Experience_yrs : float

@app.get("/")
def home():
    return FileResponse('index.html')
@app.post("/predict")
def predict(data : DeliveryInput):

    input_df = pd.DataFrame([data.model_dump()])
   
    #Predict. The model returns an array, so take the first value.
    prediction = model.predict(input_df)[0]
    final_prediction = np.exp(prediction)
    return {"delivery_time_min": round(float(final_prediction), 2)}