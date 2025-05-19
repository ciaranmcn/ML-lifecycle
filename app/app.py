from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.preprocess import preprocess_main
from app.train import train_main
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.telemetry import setup_telemetry


app = FastAPI()

@app.get("/heartbeat/{connector_id}")
def heartbeat(connector_id: str):
    return {"status": "ok", "id": connector_id}

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
    print("train() called")
    processed_path = preprocess_main(config.dataset, config.sample_size)
    result = train_main(config.model_name, processed_path)
    return {
        "status": result, 
        "preprocessed_file": processed_path
    }
