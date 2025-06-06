from temporalio.worker import Worker
from temporalio.client import Client
from workflows import FeedbackWorkflow
from activities import store_feedback_activity
async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="feedback-task-queue",
        workflows=[FeedbackWorkflow],
        activities=[store_feedback_activity]
    )

    print("Worker started. Listening for workflows...")
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())