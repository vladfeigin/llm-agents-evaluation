
import os
import pandas as pd
import json
from typing import Tuple
from promptflow.client import PFClient
from promptflow.entities import Run
from promptflow.tracing import start_trace
from multiagent_evaluation.agents.rag.rag_main import RAG
from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch
from multiagent_evaluation.utils.utils import configure_logging, configure_tracing, configure_aoai_env, load_agent_configuration

tracing_collection_name = "rag_llmops"
# Configure logging and tracing
logger = configure_logging()
tracer = configure_tracing(collection_name=tracing_collection_name)

data = "./multiagent_evaluation/agents/rag/evaluation/data.jsonl"  # Path to the data file for batch evaluation
GLOBAL_AGENT_CONFIG = None

# this function is used to run the RAG flow for batch evaluation

start_trace()

#spawn in a dedicated process to run the RAG flow
def rag_flow( session_id: str= " ", question: str = " ") -> str:
    
    # Read the environment variable, which will be a JSON string
    agent_config_str = os.getenv("GLOBAL_AGENT_CONFIG", "")
    if agent_config_str:
        agent_config = json.loads(agent_config_str)
    else:
        agent_config = {}

    logger.info(f"rag_flow:agent_config = {agent_config}")
    
    with tracer.start_as_current_span("flow::evaluation::rag_flow") as span:
        rag = RAG(agent_config)
        return rag(session_id, question)

# run the flow

def runflow(dump_output: bool = False) -> Tuple[Run, pd.DataFrame]:
    logger.info("Running the flow for batch.")
    
    with tracer.start_as_current_span("batch::evaluation::runflow") as span:
        pf = PFClient()
        try:
            base_run = pf.run(
                flow=rag_flow,
                data=data,
                description="Batch evaluation of the RAG application",
                column_mapping={
                    "session_id": "${data.session_id}",
                    "question": "${data.question}",
                    # This ground truth answer is not used in the flow
                    "answer": "${data.answer}",
                    # This context ground truth is not used in the flow
                    "context": "${data.context}",
                },
                model_config=configure_aoai_env(),
                tags={"run_configuraton": GLOBAL_AGENT_CONFIG},
                environment_variables={"GLOBAL_AGENT_CONFIG": json.dumps(GLOBAL_AGENT_CONFIG)},
                stream=True,  # To see the running progress of the flow in the console
            )
        except Exception as e:
            logger.exception(f"An error occurred during flow execution.{e}")
            print("EXCEPTION: ", e)
            raise e

        # Get run details
        details = pf.get_details(base_run)
        # if dump_to_output True, save the details to the local file called: batch_flow_output_<timestamp>.txt
        # file name must contain a current timestamp
        if dump_output:
            # timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            details.to_csv(f"batch_flow_output.txt", index=False)

        return base_run, details

# the function which runs the batch flow and then evaluates the output


def run_and_eval_flow(dump_output: bool = False):
     
    with tracer.start_as_current_span("batch::evaluation::run_and_eval_flow") as span:
        # Load the batch output from runflow
        base_run, batch_output = runflow(dump_output=dump_output)
        eval_res, eval_metrics = eval_batch(
            batch_output, dump_output=dump_output)

        # serialize the results from dictionary to json
        logger.info(
            json.dumps({
                "name": "batch-evaluation-flow-raw",
                "metadata": base_run._to_dict(),
                "result": eval_res.to_dict(orient='records')
            })
        )
        # Log the batch evaluation flow aggregated metrics
        logger.info(
            json.dumps({
                "name": "batch-evaluation-flow-metrics",
                "metadata": base_run._to_dict(),
                "result": eval_metrics.to_dict(orient='records')
            })
        )
        logger.info(">>>Batch evaluation flow completed successfully.")


if __name__ == "__main__":
    GLOBAL_AGENT_CONFIG = load_agent_configuration("agents/rag", "rag_agent_config.yaml")
    logger.info(f"GLOBAL_AGENT_CONFIG = {GLOBAL_AGENT_CONFIG}")
    run_and_eval_flow(dump_output=True)
