"""
This module provides functions to run and evaluate the agents using specified agent configuration (***_agent_config.yaml) and evaluation datasets.
You need to provide as an input evalation dataset in JSONL format, and the agent configuration file (***_agent_config.yaml) for the agent to be evaluated.
For example for RAG Agent, you need to provide rag_agent_config.yaml and data.jsonl as input.
"""
import os
from pathlib import Path
import json
from typing import Tuple, Type, Callable
import argparse
import importlib
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
            logger.error("Error processing agent: %s: %s", agent_config_file_name, e)
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
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(
            run_and_eval_flow, agent_class, eval_fn, variants_path, file, evaluation_dataset, False): file for file in files}
        # time.sleep(120)
        for future in as_completed(futures): #timeout=EVALUATION_TIMEOUT_SEC
            file = futures[future]
            try:
                evaluation_res = future.result()
                all_eval_results[file] = evaluation_res

            except Exception as e:
                logger.error("Error processing %s: %s", file, e)
                raise e
                

    return all_eval_results

# ------------------------------------------------------------------------------------------
"""
#From the root project run (single variant evaluation): 
 python -m multiagent_evaluation.agents.tools.evaluate \
  --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
  --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
  --config_dir agents/rag \
  --config_file rag_agent_config.yaml \
  --eval_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
  --dump_output \
  --mode single

# or for multiple variant evaluation:

 python -m multiagent_evaluation.agents.tools.evaluate \
  --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
  --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
  --config_dir agents/rag/evaluation/configurations/generated \
  --eval_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
  --dump_output \
  --mode multiple
"""
# ------------------------------------------------------------------------------------------




def import_from_path(full_path: str):
    """
    Dynamically import an attribute (class or function) given its full module path.
    For example: "multiagent_evaluation.agents.rag.rag_main.RAG"
    """
    module_path, attr_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation in single or multi-variant mode."
    )
    parser.add_argument(
        "--agent_class",
        type=str,
        required=True,
        help="Full module path and class name for the agent to be evaluated, e.g., multiagent_evaluation.agents.rag.rag_main.RAG"
    )
    parser.add_argument(
        "--eval_fn",
        type=str,
        required=True,
        help="Full module path and function name for evaluation function, e.g., multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Directory containing the agent configuration YAML file(s)."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Name of the agent configuration file (YAML) to use in single configuration mode."
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to the evaluation dataset in JSONL format."
    )
    parser.add_argument(
        "--dump_output",
        action="store_true",
        default=False, 
        help="If provided, the evaluation output will be dumped to a JSON files."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multiple"],
        default="single",
        help="Evaluation mode: 'single' for single agent configuration ; 'multiple' for multiple agent configurations."
    )
    args = parser.parse_args()

    # Dynamically import the agent class using the provided full path.
    agent_class = import_from_path(args.agent_class)
    eval_fn = import_from_path(args.eval_fn)

    if args.mode == "single":
        results = run_and_eval_flow(
            agent_class,
            eval_fn,
            args.config_dir,
            args.config_file,
            args.eval_dataset,
            dump_output=args.dump_output
        )
    else:  # mode == "multiple"
        results = multi_variant_evaluation(
            agent_class,
            eval_fn,
            args.config_dir,
            args.eval_dataset
        )
    print("Evaluation completed successfully. Results:", results)

if __name__ == "__main__":
    main()

