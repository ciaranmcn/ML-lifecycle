import asyncio 
from temporalio.client import Client
import time

async def main():
    client = await Client.connect("localhost:7233")
    
    handle = await client.start_workflow(
        workflow="FeedbackWorkflow",
        task_queue="feedback-task-queue",
        id=f"feedback-workflow-id-{int(time.time())}"
    )

    print("Workflow started. Sending signal...")
    await handle.signal("feedback", "Great job!")
    result = await handle.result()
    print(f"Workflow result: {result}")
if __name__ == "__main__":
    asyncio.run(main())