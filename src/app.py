from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Credit Risk ML API",
    description="API for Credit Risk ML",
    version="0.1.0",
)

class PredictionRequest(BaseModel):
    """
    Schema for prediction request.
    Adjust fields based on your model's input requirements.
    """
    input_data: List[float]
    model_name: Optional[str] = "default"

class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    prediction: float
    model_name: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Credit Risk ML API"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint to serve predictions.
    """
    # TODO: Load your model and make predictions here
    # For now, we return a dummy prediction
    
    # Example logic:
    # model = load_model(request.model_name)
    # result = model.predict(request.input_data)
    
    dummy_prediction = sum(request.input_data) / len(request.input_data) if request.input_data else 0.0
    
    return PredictionResponse(
        prediction=dummy_prediction,
        model_name=request.model_name
    )
