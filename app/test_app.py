from fastapi.testclient import TestClient
from app import app


client = TestClient(app)

def test_heartbeat():
    response = client.get("/heartbeat/test123")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "id": "test123"}

def test_predict():
    response = client.post("/predict", json={"feature1": 2.5, "feature2": 4.5})
    assert response.status_code == 200 
    response.json() == {"prediction": 7.0}

def test_train_mock(monkeypatch):
    def mock_preprocess(dataset, sample_size):
        return "mocked_output.json1"
    
    def mock_train(model_name, data_path):
        return "Mock training successful"
    
    monkeypatch.setattr("app.preprocess_main", mock_preprocess)
    monkeypatch.setattr("app.train_main", mock_train)

    response = client.post("/train", json={
        "dataset": "Open-Orca/OpenOrca",
        "sample_size": 5,
        "model_name": "tiiuae/falcon-rw-1b"
    })

    assert response.status_code == 200
    assert response.json() == {
        "status": "Mock training successful",
        "preprocessed_file": "mocked_output.jsonl"
    }