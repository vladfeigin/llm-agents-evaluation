
"""
This module provides functions to run and evaluate batch evaluations for agents using specified configurations and evaluation datasets.
"""
import os
from pathlib import Path
import json
from typing import Tuple, Type, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from multiagent_evaluation.utils.utils import configure_logging, configure_tracing, load_agent_configuration

TRACING_COLLECTION_NAME = "agent_evaluation"
EVALUATION_TIMEOUT_SEC = 300

# Configure logging and tracing
logger = configure_logging()
tracer = configure_tracing(collection_name=TRACING_COLLECTION_NAME)

# the function which runs an evaluation on test dataset for specific configiration of an agent


def run_batch(agent_class: Type, agent_config: dict, evaluation_dataset_path: str, dump_output: bool = False) -> Tuple[dict, pd.DataFrame]:
    """
    Runs a batch evaluation for a given agent on a specified agent evaluation dataset.

    Args:
        agent_class (Type): The imolementaion main class of the agent to be evaluated.
        agent_config (dict): Configuration dictionary for the agent.
        evaluation_dataset_path (str): Path to the evaluation dataset in JSONL format.
        dump_output (bool, optional): If True, the output will be saved to a JSON file, localy. Defaults to False.

    Returns:
        Tuple[dict, pd.DataFrame]: A tuple containing the agent configuration and a DataFrame with the evaluation results.

    Raises:
        ValueError: If the evaluation_dataset_path is not a valid path.
    """
    logger.info(">>>run_batch:Running batch flow for an agent evaluation.")

    # check that evaluation_dataset_path is a valid path
    if not Path(evaluation_dataset_path).exists():
        logger.error(
            "evaluation_dataset_path: %s is not a valid.", evaluation_dataset_path)
        raise ValueError("evaluation_dataset_path is not a valid path")

    outputs = []
    # load input jsonl file as pandas dataframe
    df = pd.read_json(evaluation_dataset_path, lines=True)
    for idx, row in df.iterrows():
        session_id = row['session_id']
        question = row['question']
        # Instantiate a evaluated agent from the provided agent_class with agent configuration (variant).
        agent_instance = agent_class(agent_config)
        # Call the agent instance with session_id and question.
        # TODO - currently the evaluation is oriented towards chat agents, need to generalize it!
        output = agent_instance(session_id, question)
        outputs.append(output)
    df["outputs.output"] = outputs

    if dump_output:
        # timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        df.to_json("batch_flow_output.json", index=False)
    return [agent_config, df]


@retry(
    stop=stop_after_attempt(5),                  # retry up to 5 times
    # exponential backoff starting at 2s
    wait=wait_exponential(multiplier=1, min=2),
    retry=retry_if_exception_type(Exception)     # retry on any Exception
)
def run_batch_with_retry(agent_class: Type, agent_config, evaluation_dataset, dump_output=False):
    """
    Executes a batch run of the given agent class with the specified configuration and evaluation dataset.
    Retries the batch run in case of failure.

    Args:
        agent_class (Type): The class of the agent to be evaluated.
        agent_config (dict): Configuration parameters for the agent.
        evaluation_dataset (Any): The dataset to be used for evaluation.
        dump_output (bool, optional): If True, the output will be dumped. Defaults to False.

    Returns:
        Any: The result of the batch run (evaluation data set).
    """
    return run_batch(agent_class, agent_config, evaluation_dataset, dump_output=dump_output)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2),
    retry=retry_if_exception_type(Exception)
)
def eval_batch_with_retry(eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]], batch_output, dump_output=False):
    """
    Evaluates a batch of data using the provided evaluation function. Retries the evaluation in case of failure.

    Args:
        eval_fn (Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]]): 
            The evaluation function to be applied to the batch. It should take a DataFrame and a boolean flag as input, 
            and return a tuple of two DataFrames.
        batch_output (pd.DataFrame): 
            The batch of data to be evaluated.
        dump_output (bool, optional): 
            A flag indicating whether to dump the output. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            A tuple containing two DataFrames as the result of the evaluation function.
    """
    return eval_fn(batch_output, dump_output=dump_output)


# the function which runs the batch flow and then evaluates the output
def run_and_eval_flow(agent_class: Type,
                      eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]],
                      agent_config_file_dir: str,
                      agent_config_file_name: str,
                      agent_evaluation_dataset: str,
                      dump_output: bool = False):
    """Evaluates a batch of data (evaluation data set) using the provided evaluation function.
    """

    with tracer.start_as_current_span("batch::evaluation::run_and_eval_flow") as span:
        # Load the batch output from runflow

        agent_config = load_agent_configuration(
            agent_config_file_dir, agent_config_file_name)

        try:
            run_config, batch_output = run_batch_with_retry(
                agent_class, agent_config, agent_evaluation_dataset, dump_output=dump_output)
            eval_res, eval_metrics = eval_batch_with_retry(
                eval_fn, batch_output, dump_output=dump_output)
        except Exception as e: 
            logger.error(f"Error processing agent: {agent_config_file_name}: {e}")
            raise e

        # serialize the results from dictionary to json
        logger.info(
            json.dumps({
                "name": "batch-evaluation-flow-raw",
                "metadata": run_config,  # base_run._to_dict(),
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

        return eval_res


def multi_variant_evaluation(agent_class: Type,
                             eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]],
                             variants_path: str, evaluation_dataset: str):
    """ Runs and evaluates multiple agent variants (configurations) using the provided evaluation function and evaluaiton data set."""

    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    multi_config_path = os.path.join(project_root, variants_path)

    files = [file for file in os.listdir(
        multi_config_path) if file.endswith(".yaml")]
    print(f"files = {files}")

    all_eval_results = {}
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(
            run_and_eval_flow, agent_class, eval_fn, variants_path, file, evaluation_dataset, False): file for file in files}
        # time.sleep(120)
        for future in as_completed(futures, timeout=EVALUATION_TIMEOUT_SEC):
            file = futures[future]
            try:
                evaluation_res = future.result()
                all_eval_results[file] = evaluation_res

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                raise e
                

    return all_eval_results

# ------------------------------------------------------------------------------------------
# From the root project run : python -m multiagent_evaluation.agents.tools.evaluate
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Evaluate the RAG agent
    from multiagent_evaluation.agents.rag.rag_main import RAG
    # Specific implementation of the evaluation function for the RAG agent
    from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch

    # -----------------------------------------------------------
    # single variant run:
    # -----------------------------------------------------------
    all_results = run_and_eval_flow(RAG, eval_batch, "agents/rag", "rag_agent_config.yaml",
                      "./multiagent_evaluation/agents/rag/evaluation/data.jsonl", dump_output=True)

    # -----------------------------------------------------------
    # multiple variants run:
    # -----------------------------------------------------------
    #all_results = multi_variant_evaluation(RAG, eval_batch, "agents/rag/evaluation/configurations/generated",
    # "./multiagent_evaluation/agents/rag/evaluation/data.jsonl")

    print("Evaluation completed successfully. Results>>>>>>: ", all_results)
