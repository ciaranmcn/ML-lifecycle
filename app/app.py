import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uuid
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
from temporalio.client import Client
from app.workflows import FeedbackWorkflow
from app.preprocess import preprocess_main
from app.train import train_main
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.telemetry import setup_telemetry

app = FastAPI()
temporal_client = None  

@app.on_event("startup")
async def startup_event():
    global temporal_client
    temporal_client = await Client.connect("localhost:7233")

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

@app.post("/start")
async def start_workflow():
    workflow_id = str(uuid.uuid4())
    await temporal_client.start_workflow(
        FeedbackWorkflow.run,
        id=workflow_id,
        task_queue="feedback-task-queue"
    )
    return {"workflow-id": workflow_id}

@app.post("/send")
async def send_feedback(request: Request):
    body = await request.json()
    workflow_id = body["workflow_id"]
    feedback = body["message"]

    handle = temporal_client.get_workflow_handle(workflow_id)
    await handle.signal("feedback", feedback)
    return {"status": "signal sent", "workflow-id": workflow_id}

@app.get("/result/{workflow_id}")
async def get_result(workflow_id: str):
    handle = temporal_client.get_workflow_handle(workflow_id)
    result = await handle.result()
    return {"result": result}
