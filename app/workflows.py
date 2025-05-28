from temporalio import workflow

@workflow.defn
class FeedbackWorkflow:
    @workflow.run
    async def run(self) -> str:
        msg = await workflow.wait_signal("feedback")
        return f"Got feedback: {msg}"