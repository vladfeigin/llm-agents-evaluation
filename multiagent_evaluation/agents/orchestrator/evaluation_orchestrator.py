"""
This module is responsible for orchestrating the evaluation of the agents.
Functions:
    find_optimal_agent_configuration(agent: Type, eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]], agent_config_file_dir: str, agent_config_file_name: str, evaluation_dataset: str, base_variant: str, output_dir: str = None) -> pd.DataFrame:
        Finds the optimal agent configuration by running initial evaluation, generating prompt variants, and evaluating multiple variants and other parameters.

Usage:
    Run the module using the command:
    python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator
"""
import json
from typing import Tuple, Type, Callable
import pandas as pd
from multiagent_evaluation.agents.tools.evaluate import run_and_eval_flow, multi_variant_evaluation
from multiagent_evaluation.agents.prompt_generator.prompt_generator import PromptGenerator
from multiagent_evaluation.utils.utils import load_agent_configuration
from multiagent_evaluation.agents.tools.generate_variants import generate_variants

PROMPT_GENERATOR_FOLDER = "agents/prompt_generator"
PROMPT_GENERATIR_CONFIG_FILE = "prompt_generator_config.yaml"
CONFIG_SCHEMA = "./agents/schemas/agent_config_schema.yaml"
NUMBER_OF_VARIANTS_GENERATED = 100


def find_optimal_agent_configuration(agent: Type, eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]], agent_config_file_dir: str, agent_config_file_name: str, evaluation_dataset: str, base_variant: str, output_dir: str = None):
    """
    Finds the optimal configuration for an agent by evaluating multiple prompt variants.

    Args:
        agent (Type): The agent class to be evaluated.
        eval_fn (Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]]): The evaluation function that takes a DataFrame and a boolean flag, and returns a tuple of DataFrames.
        agent_config_file_dir (str): Directory path where the agent configuration file is located.
        agent_config_file_name (str): Name of the agent configuration file.
        evaluation_dataset (str): Path to the dataset used for evaluation.
        base_variant (str): Path to the base variant JSON file.
        output_dir (str, optional): Directory where the output should be saved. Defaults to None.

    Returns:
        all_results: The results of the evaluation for all generated prompt variants.
    """

    # 1. Run initial evaluation to serve as a baseline
    eval_res = run_and_eval_flow(agent, eval_fn, agent_config_file_dir,
                                 agent_config_file_name, evaluation_dataset, dump_output=False)

    # 2. Take the inital evaluation results and create multiple prompts variants based on the initial evaluation
    pgen = PromptGenerator(load_agent_configuration(
        PROMPT_GENERATOR_FOLDER, PROMPT_GENERATIR_CONFIG_FILE))

    evaluated_agent_config = load_agent_configuration(
        agent_config_file_dir, agent_config_file_name)

    generated_prompts = pgen.generate_prompts(
        evaluated_agent_config["AgentConfiguration"]["chat_system_prompt"], eval_res)
    # load and update variants.json
    with open(base_variant, "r", encoding="utf-8") as variants_file:
        variants: dict = json.load(variants_file)
        # 3. generate the multiple variant for the agent configuration and evaluation
        generate_variants(CONFIG_SCHEMA, agent_config_file_dir,
                      agent_config_file_name, NUMBER_OF_VARIANTS_GENERATED, generated_prompts, variants, output_dir)

    # 4 run the evaluation for the multiple variants
    all_results = multi_variant_evaluation(
        agent, eval_fn, output_dir, evaluation_dataset)

    # 5. analyze the results and decide on the best agent /  best prompt or run more iteration
    # TODO analyze the results and decide on the best agent /  best prompt or run more iteration
    return all_results


if __name__ == "__main__":
    from multiagent_evaluation.agents.rag.rag_main import RAG
    from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch

    all_res = find_optimal_agent_configuration(RAG, eval_batch, "agents/rag", "rag_agent_config.yaml", "./multiagent_evaluation/agents/rag/evaluation/data.jsonl",
                                               "./multiagent_evaluation/agents/rag/variants.json",  "./agents/rag/evaluation/configurations/generated")
    print("Evaluation completed successfully. Results>>>>>>: ", all_res)
