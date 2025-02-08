# this module is responsible for orchestrating the evaluation of the agents
# it first calls run_and_eval_flow which runs the agent and evaluates the output
# user add the task in natural language and the agent will be evaluated based on the task
# python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator

from typing import Tuple, Type, Callable
import pandas as pd
from multiagent_evaluation.agents.tools.evaluate import run_and_eval_flow, multi_variant_evaluation
from multiagent_evaluation.agents.prompt_generator.prompt_generator import PromptGenerator
from multiagent_evaluation.utils.utils import load_agent_configuration
from multiagent_evaluation.agents.tools.generate_variants import generate_variants

PROMPT_GENERATOR_FOLDER = "agents/prompt_generator"
PROMPT_GENERATIR_CONFIG_FILE = "prompt_generator_config.yaml"
CONFIG_SCHEMA = "./agents/schemas/agent_config_schema.yaml"


def find_optimal_agent_configuration(agent: Type, eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]], agent_config_file_dir: str, agent_config_file_name: str, evaluation_dataset: str, base_variant: str, output_dir: str = None):

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
    with open(base_variant, "r", encoding="utf-8") as variants:
        #variants: dict = json.load(open(base_variant, "r", encoding="utf-8"))
        # 3. generate the multiple variantshi i can use
        generate_variants(CONFIG_SCHEMA, agent_config_file_dir,
                      agent_config_file_name, 100, generated_prompts, variants, output_dir)

    # 4 run the evaluation for the multiple variants
    all_results = multi_variant_evaluation(
        agent, eval_fn, output_dir, evaluation_dataset)

    # 5. analyze the results and decide on the best agent /  best prompt or run more iteration
    # ...
    return all_results


if __name__ == "__main__":
    from multiagent_evaluation.agents.rag.rag_main import RAG
    from multiagent_evaluation.agents.rag.evaluation.evaluation_implementation import eval_batch

    all_res = find_optimal_agent_configuration(RAG, eval_batch, "agents/rag", "rag_agent_config.yaml", "./multiagent_evaluation/agents/rag/evaluation/data.jsonl",
                                               "./multiagent_evaluation/agents/rag/variants.json",  "./agents/rag/evaluation/configurations/generated")
    print("Evaluation completed successfully. Results>>>>>>: ", all_res)
