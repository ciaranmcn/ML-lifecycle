from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from preprocess import preprocess_main
from train import train_main
from fastapi.testclient import TestClient
app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(data: InputData):
    result = data.feature1 + data.feature2
    return {"prediction": result}

class FullTrainConfig(BaseModel):
    model_name: str
    dataset: str
    sample_size: int

@app.post("/train")
def train(config: FullTrainConfig):
    processed_path = preprocess_main(config.dataset, config.sample_size)
    result = train_main(config.model_name, processed_path)
    return {
        "status": result, 
        "preprocessed_file": processed_path
    }

@app.get("/heartbeat/{connector_id}")
def heartbeat(connector_id: str):
    return {"status": "ok", "id": connector_id}

def test_read_main():
    repsonse = client.get