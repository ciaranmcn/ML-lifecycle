from temporalio import workflow

@workflow.defn
class FeedbackWorkflow:
    def __init__(self) -> None:
        self.feedback_msg = None
        
    @workflow.signal
    async def feedback(self, value: str):
        workflow.logger.info(f"Signal recieved: {value}")
        self.feedback_msg = value

    @workflow.run
    async def run(self) -> str:
        await workflow.wait_condition(lambda: self.feedback_msg is not None)
        return f"Got feedback: {self.feedback_msg}"

