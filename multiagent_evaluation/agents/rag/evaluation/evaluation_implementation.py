import os
import pandas as pd
from dotenv import load_dotenv
from multiagent_evaluation.utils.utils import configure_logging
from azure.ai.evaluation import (
    RelevanceEvaluator,
    GroundednessEvaluator,
    SimilarityEvaluator,
    CoherenceEvaluator,
)

# Load environment variables
load_dotenv()

# Configure logging
logger = configure_logging()

# Get LLM configuration
model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_EVALUATION_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_KEY"),
    "azure_deployment": os.getenv("AZURE_OPENAI_EVALUATION_DEPLOYMENT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
}


# Initialize dictionary with evaluators
evaluators = {
    "relevance": RelevanceEvaluator(model_config),
    "groundedness": GroundednessEvaluator(model_config),
    "similarity": SimilarityEvaluator(model_config),
    "coherence": CoherenceEvaluator(model_config),
}

# Function to verify prompt file paths


def verify_prompty_files(evaluators: dict):
    for name, evaluator in evaluators.items():
        try:
            prompty_file = evaluator.PROMPTY_FILE
            if os.path.exists(prompty_file):
                logger.info(
                    f"Evaluator '{name}' is using existing prompty file: {prompty_file}")
            else:
                logger.error(
                    f"Evaluator '{name}' is using missing prompty file: {prompty_file}")
        except AttributeError:
            logger.warning(
                f"Evaluator '{name}' does not have a PROMPTY_FILE attribute.")

# Individual evaluator functions


def relevance(row):
    return evaluators["relevance"](
        response=row["outputs.output"],
        context=row["context"],
        query=row["question"]
    )["gpt_relevance"]


def groundedness(row):
    return evaluators["groundedness"](
        response=row["outputs.output"],
        context=row["context"]
    )["gpt_groundedness"]


def similarity(row):
    return evaluators["similarity"](
        response=row["outputs.output"],
        ground_truth=row["answer"],
        query=row["question"]
    )["gpt_similarity"]


def coherence(row):
    return evaluators["coherence"](
        response=row["outputs.output"],
        query=row["question"]
    )["gpt_coherence"]


# Mapping of evaluator functions for easy access
evaluator_funcs = {
    "relevance": relevance,
    "groundedness": groundedness,
    "similarity": similarity,
    "coherence": coherence,
}

# Calculate average score


def calc_score(scores):
    return sum(scores) / len(scores) if scores else 0

# Calculate overall scores and return DataFrame


def calculate_overall_score(scores):
    calc_scores = {"metric": [], "score": []}
    for key, values in scores.items():
        calc_scores["metric"].append(key)
        calc_scores["score"].append(calc_score(values))
    return pd.DataFrame(calc_scores)

# Evaluate all scores and return detailed and summary results


def eval_batch(batch_output: pd.DataFrame, dump_output: bool = False):
    # Verify prompt files
    # verify_prompty_files(evaluators)

    # Initialize result DataFrame
    eval_res = pd.DataFrame(columns=[])

    scores = {key: [] for key in evaluator_funcs.keys()}

    try:
        for _, row in batch_output.iterrows():
            new_row = row.to_dict()

            # Calculate scores and append to new_row and scores dictionary
            for eval_name, func in evaluator_funcs.items():
                score = func(row)
                new_row[eval_name] = score
                scores[eval_name].append(score)

            # Append to result DataFrame
            eval_res = pd.concat(
                [eval_res, pd.DataFrame([new_row])], ignore_index=True)

        # Calculate overall average scores
        eval_metrics = calculate_overall_score(scores)
    except Exception as e:
        logger.exception(f"An error occurred during batch evaluation: {e}")
        print("EXCEPTION in Evaluaitons: ", e)
        raise e

    # Optional: Dump results to files
    if dump_output:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        eval_res.to_json(
            f"eval_results_{timestamp}.json", orient='records', lines=True)
        eval_metrics.to_json(
            f"batch_eval_output_{timestamp}.json", orient='records', lines=True)

    return eval_res, eval_metrics
