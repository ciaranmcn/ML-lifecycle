from temporalio import workflow
from opentelemetry import trace 
tracer = trace.get_tracer(__name__)

@workflow.defn
class FeedbackWorkflow:
    def __init__(self) -> None:
        self.feedback_msg = None
        self._cancelled = False
        
    @workflow.signal
    async def feedback(self, msg: str):
        workflow.logger.info(f"Signal recieved: {msg}")
        self.feedback_msg = msg

    @workflow.signal
    async def cancel(self):
        self._cancelled = True
        
    @workflow.signal
    async def update_feedback(self, new_msg):
        self.feedback_msg =f"(Updated) {new_msg}"
        
    @workflow.run
    async def run(self) -> str:
        with tracer.start_as_current_span("workflow_run"):
            workflow.logger.info("Waiting for feedback...")
            await workflow.wait_condition(lambda: self.feedback_msg is not None or self._cancelled)
            
            if self._cancelled:
                return "Workflow was cancelled"

            workflow.logger.info(f"Got feedback: {self.feedback_msg}")
            
            await workflow.execute_activity(
                store_feedback_activity, self.feedback_msg,
                start_to_close_timeout=timedelta(seconds=10)
            )
            return f"Final response: {self.feedback_msg}"



