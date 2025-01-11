
import os
import pandas as pd
import json
from multiprocessing import Process
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Tuple
from promptflow.client import PFClient
# from promptflow.azure import PFClient
# from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from promptflow.entities import Run
from promptflow.tracing import start_trace
from multiagent_evaluation.agents.rag.rag_main import RAG
from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch
from multiagent_evaluation.utils.utils import configure_logging, configure_tracing, configure_aoai_env, load_agent_configuration


tracing_collection_name = "rag_llmops"
# Configure logging and tracing
logger = configure_logging()
tracer = configure_tracing(collection_name=tracing_collection_name)

# Path to the data file for batch evaluation
data = "./multiagent_evaluation/agents/rag/evaluation/data.jsonl"
GLOBAL_AGENT_CONFIG = None

# this function is used to run the RAG flow for batch evaluation

start_trace()

""" 
def init():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential()
    return credential
"""

# spawn in a dedicated process to run the RAG flow
"""
def rag_flow(session_id: str = " ", question: str = " ") -> str:

    with tracer.start_as_current_span("flow::evaluation::rag_flow") as span:
        rag = RAG()
        return rag(session_id, question)
"""

# run the flow

def runflow(agent_config, dump_output: bool = False) -> Tuple[Run, pd.DataFrame]:
    logger.info("Running the flow for batch.")

    with tracer.start_as_current_span("batch::evaluation::runflow") as span:
        pf = PFClient(config={'trace.destination': "Local"})
        # Connect to the workspace if Azure
        # pf = PFClient.from_config(credential=credential)
        try:
            base_run = pf.run(
                flow="multiagent_evaluation/agents/rag",   # rag_flow,
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
                tags={"run_configuraton": agent_config},
                environment_variables={
                    "aoai_config": json.dumps(GLOBAL_AGENT_CONFIG)},
                init={"rag_config": json.dumps(agent_config)},
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


def run_and_eval_flow(agent_config, dump_output: bool = False):

    with tracer.start_as_current_span("batch::evaluation::run_and_eval_flow") as span:
        # Load the batch output from runflow
        base_run, batch_output = runflow(agent_config, dump_output=dump_output)
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


def process_config(file: str, dump_output: bool = False):
    logger.info(f"Loading configuration from {file}")
    agent_config = load_agent_configuration(
        "agents/rag/evaluation/configurations/generated", file)
    logger.info(f"agent_config = {agent_config}")
    run_and_eval_flow(agent_config, dump_output=dump_output)


##------------------------------------------------------------------------------------------
# From the root project run : python -m multiagent_evaluation.agents.rag.evaluation.evaluate
##------------------------------------------------------------------------------------------

"""
#----------------------------------------------------------------------------------------------------------------------------
# Multi variant run: running multiple batches of dataset, each with diferent parameters to find best performing configuration
#----------------------------------------------------------------------------------------------------------------------------

def main():

    # init aoai global parameters
    GLOBAL_AGENT_CONFIG = configure_aoai_env()

    config_dir = "./multiagent_evaluation/agents/rag/evaluation/configurations/generated/"
    files = [file for file in os.listdir(config_dir) if file.endswith(".yaml")]
    print(f"files = {files}")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            process_config, file, False): file for file in files}
        time.sleep(120)
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()

"""
#-------------------------------------------------------------------------------------------
#Single variant run  
#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    agent_config = load_agent_configuration("agents/rag", "rag_agent_config.yaml")
    logger.info(f"GLOBAL_AGENT_CONFIG = {agent_config}")
    print(f"GLOBAL_AGENT_CONFIG = {agent_config}")
    run_and_eval_flow(agent_config, dump_output=True)
    

