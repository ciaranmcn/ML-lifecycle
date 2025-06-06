from temporalio import activity

@activity.defn
async def store_feedback_activity(msg:str) -> None:
    activity.logger.info(f"[Activity] Storing feedback: {msg}")
