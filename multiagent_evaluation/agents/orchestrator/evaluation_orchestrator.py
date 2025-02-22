"""
This module is responsible for orchestrating the evaluation of the agents.
Functions:
    find_optimal_agent_configuration(
        agent: Type, 
        eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]], 
        agent_config_file_dir: str, 
        agent_config_file_name: str, 
        evaluation_dataset: str, 
        base_variant: str, 
        output_dir: Optional[str] = None
    ) -> str:
        Finds the optimal agent configuration by running initial evaluation, generating prompt variants,
        and evaluating multiple variants and other parameters.

Usage:
    Run the module using the command:
    python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator
"""
import os
import json
import argparse
import importlib
# CHANGE: Added Optional, Any, Dict type hints
from typing import Tuple, Type, Callable, Optional, Any, Dict
import pandas as pd
from langchain_core.prompts import PromptTemplate
from multiagent_evaluation.agents.tools.evaluate import run_and_eval_flow, multi_variant_evaluation
from multiagent_evaluation.agents.prompt_generator.prompt_generator import PromptGenerator
from multiagent_evaluation.utils.utils import load_agent_configuration, configure_tracing, configure_logging
from multiagent_evaluation.agents.tools.generate_variants import generate_variants
from multiagent_evaluation.aimodel.ai_model import AIModel

# Configure tracing and logging
logger = configure_logging()
tracer = configure_tracing(__file__)

# Constants
PROMPT_GENERATOR_FOLDER: str = "agents/prompt_generator"
PROMPT_GENERATOR_CONFIG_FILE: str = "prompt_generator_config.yaml"
CONFIG_SCHEMA: str = "./agents/schemas/agent_config_schema.yaml"
NUMBER_OF_VARIANTS_GENERATED: int = 10 


class Orchestrator:
    def __init__(self, agent_config: Optional[Dict] = None) -> None:
        logger.info("Orchestrator.Initializing...")
        with tracer.start_as_current_span("PromptGenerator.__init__span") as span:
            try:
                if agent_config is None:
                    # load configuration from default variant yaml
                    logger.info(
                        "Orchestrator.__init__: agent_config is empty, loading default configuration")
                    agent_config = load_agent_configuration(
                        "agents/orchestrator", "evaluation_orchestrator_agent_config.yaml")

                api_key = os.getenv("AZURE_OPENAI_KEY")
                # check if agent config and api_key are provided
                if agent_config is None or api_key is None:
                    logger.error("Agent config and api_key are required")
                    raise ValueError("Agent config and api_key are required")

                self.agent_config = agent_config
                logger.info("__init__.agent_config = %s", agent_config)
                span.set_attribute("orchestrator_agent_config", agent_config)

                # init the AIModel class enveloping a LLM model
                self.aimodel = AIModel(
                    azure_deployment=self.agent_config["AgentConfiguration"]["deployment"]["name"],
                    openai_api_version=self.agent_config["AgentConfiguration"]["deployment"]["openai_api_version"],
                    azure_endpoint=self.agent_config["AgentConfiguration"]["deployment"]["endpoint"],
                    api_key=api_key,
                    model_parameters={
                        "temperature": self.agent_config["AgentConfiguration"]["model_parameters"]["temperature"]
                    }
                )
            except Exception as e:
                logger.exception("Error initializing Orchestrator: %s", e)

    def find_optimal_agent_configuration(
        self,
        agent: Type,
        eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]],
        agent_config_file_dir: str,
        agent_config_file_name: str,
        evaluation_dataset: str,
        base_variant: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Finds the optimal configuration for an agent by evaluating multiple prompt variants.
        Returns a JSON string representing the evaluation results.
        """

        # 1. Run initial evaluation to serve as a baseline
        eval_res = run_and_eval_flow(
            agent, eval_fn, agent_config_file_dir,
            agent_config_file_name, evaluation_dataset, dump_output=False
        )

        # 2. Generate prompt variants based on the initial evaluation
        pgen = PromptGenerator(load_agent_configuration(
            PROMPT_GENERATOR_FOLDER, PROMPT_GENERATOR_CONFIG_FILE))
        evaluated_agent_config = load_agent_configuration(
            agent_config_file_dir, agent_config_file_name)
        generated_prompts = pgen.generate_prompts(
            evaluated_agent_config["AgentConfiguration"]["chat_system_prompt"], eval_res
        )

        try:
            # 3. Load the base variant definitions and generate multiple variants for evaluation
            with open(base_variant, "r", encoding="utf-8") as variants_file:
                variants: dict = json.load(variants_file)
            generate_variants(
                CONFIG_SCHEMA,
                agent_config_file_dir,
                agent_config_file_name,
                NUMBER_OF_VARIANTS_GENERATED,
                generated_prompts,
                variants,
                output_dir
            )
        except Exception as e:
            logger.exception("Error generating variants: %s", e)
            raise

        # 4. Run the evaluation for the multiple variants
        all_results = multi_variant_evaluation(
            agent, eval_fn, output_dir, evaluation_dataset)

        logger.info(
            "orchestrator::find_optimal_agent_configuration#all_results = %s", all_results)

        # 5. Convert results to JSON using a custom serializer
        evaluation_results = json.dumps(
            all_results, default=self.__serializer__)

        return evaluation_results

    def analyze(self, evaluation_results: str) -> Any:
        """
        Analyze the evaluation results and decide on the best agent/prompt.
        """
        logger.info("Orchestrator.analyze")
        try:
            with tracer.start_as_current_span("Orchestrator.analyzer") as span:
                prompt_template = PromptTemplate.from_template(
                    self.agent_config["AgentConfiguration"]["system_prompt"]
                )
                span.set_attribute("evaluation_results", evaluation_results)
                chain = prompt_template | self.aimodel.llm()
                return chain.invoke({"evaluation_results": evaluation_results})
        except Exception as e:
            logger.exception("Error during analysis: %s", e)
            raise

    def __serializer__(self, obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        raise TypeError("Type not serializable")


def import_from_path(full_path: str) -> Any:
    """
    Dynamically import a class or function from its full module path.
    """
    try:
        module_path, attr_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except Exception as e:
        logger.exception("Error importing %s: %s", full_path, e)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agent evaluation orchestrator to find optimal configuration."
    )
    parser.add_argument(
        "--agent_class",
        type=str,
        required=True,
        help="Full module path and class name for the agent (e.g., multiagent_evaluation.agents.rag.rag_main.RAG)."
    )
    parser.add_argument(
        "--eval_fn",
        type=str,
        required=True,
        help="Full module path and function name for evaluation (e.g., multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch)."
    )
    parser.add_argument(
        "--agent_config_file_dir",
        type=str,
        required=True,
        help="Directory containing the agent configuration file."
    )
    parser.add_argument(
        "--agent_config_file_name",
        type=str,
        required=True,
        help="Name of the agent configuration file."
    )
    parser.add_argument(
        "--evaluation_dataset",
        type=str,
        required=True,
        help="Path to the dataset for evaluation."
    )
    parser.add_argument(
        "--base_variant",
        type=str,
        required=True,
        help="Path to the variant definitions JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the generated configurations (variants) will be saved."
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        # Dynamically load agent class and evaluation function.
        agent_class = import_from_path(args.agent_class)
        eval_fn = import_from_path(args.eval_fn)
        orchestrator = Orchestrator()

        evaluation_results = orchestrator.find_optimal_agent_configuration(
            agent=agent_class,
            eval_fn=eval_fn,
            agent_config_file_dir=args.agent_config_file_dir,
            agent_config_file_name=args.agent_config_file_name,
            evaluation_dataset=args.evaluation_dataset,
            base_variant=args.base_variant,
            output_dir=args.output_dir
        )

        answer = orchestrator.analyze(evaluation_results)
        logger.info("Best configuration: %s", answer)
        print(f"Best configuration = {answer}")
    except Exception as e:
        logger.exception("Error in main execution: %s", e)
        print("An error occurred. Check logs for details.")


if __name__ == "__main__":
    main()


""" Example of running the orchestrator for RAG agent:

python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator \
    --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
    --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
    --agent_config_file_dir agents/rag \
    --agent_config_file_name rag_agent_config.yaml \
    --evaluation_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
    --base_variant ./multiagent_evaluation/agents/rag/variants.json \
    --output_dir ./agents/rag/evaluation/configurations/generated


"""
