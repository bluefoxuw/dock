from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import uvicorn

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal Length")
    sepal_width: float = Field(gt=0, description="Sepal Width")
    petal_length: float = Field(gt=0, description="Petal Length")
    petal_width: float = Field(gt=0, description="Petal Width")

# Define output schema
class PredictionResponse(BaseModel):
    species: str
    probability: float

# Initialize FastAPI app
app = FastAPI(title="ML Model API")

@app.post("/predict/")
def predict(features: IrisFeatures):
    print("Received request:", features.dict())  # Check if request data is received

    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    
    print("Data for prediction:", data)

    species_id = model.predict(data)[0]  # Ensure this runs
    print("Prediction result:", species_id)

    proba = max(model.predict_proba(data)[0])
    print("Probability:", proba)

    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    species = species_map[species_id]

    return PredictionResponse(species=species, probability=proba)

@app.get("/")
def read_root():
    return "NIGGA COME"

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)