from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # To load the model
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model (rf.pkl)
model = joblib.load('random_forest_model.pkl')

# Define the input data structure using Pydantic
class PredictRequest(BaseModel):
    Hydraulic_Pressure: float
    Coolant_Pressure: float
    Air_System_Pressure: float
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float
    Spindle_Bearing_Temperature: float
    Spindle_Vibration: float
    Tool_Vibration: float
    Spindle_Speed: float
    Voltage: float
    Torque: float
    Cutting: float

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    # Extract features from the request
    features = [
        request.Hydraulic_Pressure,
        request.Coolant_Pressure,
        request.Air_System_Pressure,
        request.Coolant_Temperature,
        request.Hydraulic_Oil_Temperature,
        request.Spindle_Bearing_Temperature,
        request.Spindle_Vibration,
        request.Tool_Vibration,
        request.Spindle_Speed,
        request.Voltage,
        request.Torque,
        request.Cutting
    ]

    # Convert the features into a numpy array
    input_features = np.array([features])

    # Predict downtime using the trained model
    prediction = model.predict(input_features)

    # Return the prediction result as "Yes" or "No"
    downtime = "Yes" if prediction[0] == 1 else "No"
    return {"Downtime": downtime}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
