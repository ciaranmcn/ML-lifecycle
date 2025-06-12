import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uuid
from uuid import UUID
from typing import List
from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from temporalio.client import Client
from app.workflows import FeedbackWorkflow
from app.preprocess import preprocess_main
from app.train import train_main
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.telemetry import setup_telemetry

app = FastAPI()
tracer = setup_telemetry()
FastAPIInstrumentor.instrument_app(app)
temporal_client = None  


@app.on_event("startup")
async def startup_event():
    global temporal_client
    temporal_client = await Client.connect("localhost:7233")

@app.post("/start/{workflow_id}")
async def start_workflow(workflow_id: UUID):
    with tracer.start_as_current_span("start_workflow"):
        try: 
            await temporal_client.start_workflow(
                FeedbackWorkflow.run,
                id=workflow_id,
                task_queue="feedback-task-queue"
            )
            return {"workflow-id": workflow_id}
        except Exception as e:
            return {"exception": str(e)}
    
# intercept the signal, signal handler
@app.post("/send/{workflow_id}")
async def send_feedback(workflow_id: UUID, feedback:str = Body(..., embed=True)):
    with tracer.start_as_current_span("send_feedback"):
        handle = temporal_client.get_workflow_handle(workflow_id)
        await handle.signal("feedback", feedback)
        return {"status": "signal sent", "workflow-id": workflow_id}
    
# to retireve result : workdlow id as a parameter on all
@app.get("/result/{workflow_id}")
async def get_result(workflow_id: UUID):
    with tracer.start_as_current_span("get_result"):
        handle = temporal_client.get_workflow_handle(workflow_id)
        result = await handle.result()
        return {"result": result}


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
