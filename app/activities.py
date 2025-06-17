from datetime import datetime
from temporalio import activity
from app.db import get_connection

@activity.defn
async def store_feedback_activity(feedback: str) -> None:
    print(f"Storing feedback: {feedback}")
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("INSERT INTO feedback (message) VALUES (%s)", (feedback,))
        conn.commit()
    finally:
        conn.close()
