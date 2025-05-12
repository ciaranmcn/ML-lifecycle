import json
from datetime import datetime 
from pathlib import Path

LOG_FILE = Path("logs.jsonl")

def log_result(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
    print(f"Logged to {LOG_FILE}")
                