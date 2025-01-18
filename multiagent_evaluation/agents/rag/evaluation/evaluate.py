
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_evaluation.agents.rag.rag_main import RAG
from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch
from multiagent_evaluation.utils.utils import configure_logging, configure_tracing, configure_aoai_env, load_agent_configuration


tracing_collection_name = "rag_llmops"
# Configure logging and tracing
logger = configure_logging()
tracer = configure_tracing(collection_name=tracing_collection_name)

# Path to the data file for batch evaluation
data = "./multiagent_evaluation/agents/rag/evaluation/data.jsonl"

# the function which runs the batch flow
def run_batch(agent_config: dict, dump_output: bool = False) -> Tuple[dict, pd.DataFrame]:

    logger.info(">>>run_batch:Running batch flow for RAG evaluation.")
    
    # check that evaluation_dataset_path is a valid path
    if not os.path.exists(data):
        logger.error("evaluation_dataset_path is not a valid.")
        raise ValueError("evaluation_dataset_path is not a valid path")

    outputs = []
    # load input jsonl file as pandas dataframe
    df = pd.read_json(data, lines=True)
    for idx, row in df.iterrows():
        session_id = row['session_id']
        question = row['question']
        rag = RAG(agent_config)
        outputs.append(rag(session_id, question))
    df["outputs.output"] = outputs

    if dump_output:
        # timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        df.to_json(f"batch_flow_output.json", index=False)

    return [agent_config, df]


@retry(
    stop=stop_after_attempt(5),                  # retry up to 5 times
    # exponential backoff starting at 2s
    wait=wait_exponential(multiplier=1, min=2),
    retry=retry_if_exception_type(Exception)     # retry on any Exception
)
def run_batch_with_retry(agent_config, dump_output=False):
    return run_batch(agent_config, dump_output=dump_output)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2),
    retry=retry_if_exception_type(Exception)
)
def eval_batch_with_retry(batch_output, dump_output=False):
    return eval_batch(batch_output, dump_output=dump_output)


# the function which runs the batch flow and then evaluates the output
def run_and_eval_flow(config_file_dir:str ,config_file_name: str, dump_output: bool = False):

    with tracer.start_as_current_span("batch::evaluation::run_and_eval_flow") as span:
        # Load the batch output from runflow

        agent_config = load_agent_configuration(config_file_dir, config_file_name)

        run_config, batch_output = run_batch_with_retry(
            agent_config, dump_output=dump_output)

        eval_res, eval_metrics = eval_batch_with_retry(
            batch_output, dump_output=dump_output)

        # serialize the results from dictionary to json
        logger.info(
            json.dumps({
                "name": "batch-evaluation-flow-raw",
                "metadata":  run_config,  # base_run._to_dict(),
                "result": eval_res.to_dict(orient='records')
            })
        )
        # Log the batch evaluation flow aggregated metrics
        logger.info(
            json.dumps({
                "name": "batch-evaluation-flow-metrics",
                "metadata": run_config,  # base_run._to_dict(),
                "result": eval_metrics.to_dict(orient='records')
            })
        )
        logger.info(">>>Batch evaluation flow completed successfully.")

# ------------------------------------------------------------------------------------------
# From the root project run : python -m multiagent_evaluation.agents.rag.evaluation.evaluate
# ------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Multi variant run: running multiple batches of dataset, each with diferent parameters to find best performing configuration
# ----------------------------------------------------------------------------------------------------------------------------

def main():

    config_dir = "./multiagent_evaluation/agents/rag/evaluation/configurations/generated/"
    files = [file for file in os.listdir(config_dir) if file.endswith(".yaml")]
    print(f"files = {files}")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            run_and_eval_flow, "agents/rag/evaluation/configurations/generated", file, False): file for file in files}
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
    run_and_eval_flow( "agents/rag", "rag_agent_config.yaml", dump_output=True )
"""
