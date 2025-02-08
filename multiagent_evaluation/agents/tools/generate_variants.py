#!/usr/bin/env python3
"""
generate-variants.py

Generates multiple YAML configuration files (named rag_agent_config_*.yaml)
based on:
  - a base YAML template (agent_config_schema.yaml),
  - a JSON file (variants.json) that specifies possible values or ranges of values,
    plus a new "active" attribute to skip certain items if active:false.    
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import List

import yaml
from jsonschema import validate
from multiagent_evaluation.utils.utils import load_agent_configuration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple YAML files from a base YAML schema and variants."
    )
    parser.add_argument(
        "variants_json",
        type=str,
        help="Path to the variants.json file with the possible or range/set values."
    )
    parser.add_argument(
        "schema_yaml",
        type=str,
        help="Path to the agent_config_schema.yaml file describing the schema."
    )
    parser.add_argument(
        "--base_config_yaml",
        type=str,
        default=None,
        help="Optional path to a base rag_agent_config.yaml template. "
             "If not provided, a minimal base config is constructed."
    )
    parser.add_argument(
        "--max_variants",
        type=int,
        default=100,
        help="Maximum number of variants to generate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated YAML files."
    )
    return parser.parse_args()

def load_schema(schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_variants(variants_path: str):
    with open(variants_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_base_config(base_config_path: str):
    """
    Load and return the base rag_agent_config.yaml as a Python dict.
    If none provided, return a minimal base that aligns with the schema requirements.
    """
    if base_config_path and os.path.isfile(base_config_path):
        with open(base_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        # A minimal base config matching your schema fields
        return {
            "AgentConfiguration": {
                "agent_name": "rag_agent",
                "description": "RAG Agent configuration",
                "config_version": "1.2",
                "application_version": "1.0",
                "application_name": "rag_llmops_workshop",
                "deployment": {
                    "model_name": "gpt-4o-mini",
                    "model_version": "2024-08-01",
                    "name": "gpt-4o-mini",
                    "endpoint": "https://example.com/gpt-4o-mini",
                    "openai_api_version": "2024-10-01-preview",
                },
                "model_parameters": {
                    "temperature": 0.0,
                    "seed": 42
                },
                "retrieval": {
                    "parameters": {
                        "search_type": "hybrid",
                        "top_k": 5,
                        "index_name": "micirosoft-tech-stack-index-0",
                        "index_semantic_configuration_name": "vector-llmops-workshop-index-semantic-configuration",
                    },
                    "deployment": {
                        "model_name": "text-embedding-ada-002",
                        "name": "text-embedding-ada-002",
                        "endpoint": "https://example.com/text-embedding-ada-002",
                        "openai_api_version": "2024-10-01-preview",
                    },
                },
                "intent_system_prompt": (
                    "Your task is to extract the user’s intent by reformulating their latest question..."
                ),
                "chat_system_prompt": (
                    "You are a knowledgeable assistant specializing only in technology domain..."
                ),
                "human_template": "question: {input}"
            }
        }

def float_range(start, end, step):
    """Generate a range of floats [start, end] with the given step."""
    vals = []
    current = float(start)
    while current <= end + 1e-9:
        vals.append(round(current, 5))
        current += step
    return vals

def int_range(start, end, step):
    """Generate a range of integers [start, end] with the given step."""
    return list(range(start, end + 1, step))

def is_active(d):
    """
    Return True if dictionary d is "active" (either no 'active' key, or it is true).
    Return False if d['active'] is false or "false" or anything similar.
    """
    if "active" not in d:
        return True
    val = d["active"]
    # Accept booleans or strings
    if val in [True, "true", "True", 1, "1"]:
        return True
    return False

def parse_param_info(param_info):
    """
    param_info might be something like:
      { "range":[0.0,1.0], "step":0.5, "default":0.0, "active": "true" }
    We interpret "range"/"step"/"set"/"default", ignoring leftover keys.
    If not recognized or not active, return an empty list.
    """
    # If not active, skip
    if not is_active(param_info):
        return []

    # 1) "set": [values]
    if "set" in param_info and isinstance(param_info["set"], list):
        return param_info["set"]

    # 2) "range": [start,end], plus "step"
    if "range" in param_info and "step" in param_info:
        rng = param_info["range"]
        if isinstance(rng, list) and len(rng) == 2:
            start, end = rng
            step = param_info["step"]
            # float or int?
            if any(isinstance(x, float) for x in [start, end, step]):
                return float_range(float(start), float(end), float(step))
            else:
                return int_range(int(start), int(end), int(step))

    # 3) "default": single value
    if "default" in param_info:
        return [param_info["default"]]

    # If none recognized, return empty
    return []

def list_of_dicts_to_param_dict(list_of_dicts):
    """
    The user gave something like model_parameters: [
       { "name": "temperature", "range": [0.0,1.0], "step":0.5, "active": "true" },
       { "name": "top_p", "range": [0.1,0.9], "step":0.5, "active": false },
       ...
    ]
    We only include items that are is_active(...) => True
    We produce a dictionary:
      "temperature": { "range":[0.0,1.0], "step":0.5 },
      etc.
    """
    output = {}
    for d in list_of_dicts:
        if not is_active(d):
            continue
        if "name" in d:
            param_name = d["name"]
            # Copy over recognized fields (range, step, set, default), ignoring leftover
            # We'll just keep the entire d minus 'name' and 'active'
            param_def = {}
            for k, v in d.items():
                if k not in ["name", "active"]:
                    param_def[k] = v
            output[param_name] = param_def
    return output

def build_combinations_for_section(section_variants):
    """
    If section_variants is a dict like:
      {
        "temperature": {"range":[0.0,1.0], "step":0.5},
        "top_p":       {"range":[0.1,0.9], "step":0.5}
      }
    We parse each param's info with parse_param_info => list of scalars, then do cartesian product.
    """
    from itertools import product

    if not isinstance(section_variants, dict) or not section_variants:
        return [{}]

    param_names = []
    value_lists = []

    for param_name, param_info in section_variants.items():
        possible_vals = parse_param_info(param_info)
        if possible_vals:
            param_names.append(param_name)
            value_lists.append(possible_vals)

    # If no recognized params or all were inactive => single empty combo
    if not param_names:
        return [{}]

    combos = []
    for combo in product(*value_lists):
        cdict = {}
        for i, name in enumerate(param_names):
            cdict[name] = combo[i]
        combos.append(cdict)

    return combos

def build_value_combinations(variants, key_name):
    """
    For a top-level or sub-level key (like "model_parameters", or "retrieval.parameters"),
    the user might have:
      - a list of dicts (each presumably with "name" and maybe "active"),
      - or a dict of param_name->param_info,
      - or a scalar
    We unify it by:
      - If it's a list of dicts, filter out inactive ones, then convert to a single dict
      - Then pass to build_combinations_for_section
    """
    if key_name not in variants:
        return [{}]

    data = variants[key_name]

    # 1) If it's a list of dicts
    if isinstance(data, list) and len(data) > 0 and all(isinstance(x, dict) for x in data):
        param_dict = list_of_dicts_to_param_dict(data)
        return build_combinations_for_section(param_dict)

    # 2) If it's a dict
    if isinstance(data, dict):
        return build_combinations_for_section(data)

    # 3) Otherwise (scalar or something else)
    return [{}]


def set_prompt_variants(variants:dict, prompt_variants: List[str]):
    #from prompts dictionary retrieve the list of prompts and add them to the variants
    prompts =[]

    for prompt in prompt_variants["prompts"]:
        prompts.append( prompt["prompt"] + "  \n<context>\n{context}\n</context>")
    variants["chat_system_prompt"] = prompts
    return variants
    

def generate_variants( schema_yaml:str, agent_folder: str, agent_config_file: str , max_variants:int, prompt_variants: List[str], variants:dict, output_dir:str):
   
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    schema_yaml = os.path.join(project_root, schema_yaml)
    output_dir = os.path.join(project_root, output_dir)
    
    # 0) Update variants dictionary
    variants: dict = set_prompt_variants(variants, prompt_variants)
    
    # 1) Load schema, variants, base
    schema = load_schema(schema_yaml)
    #base_config = load_base_config(base_config_yaml)
    base_config = load_agent_configuration(agent_folder, agent_config_file)

    # 2) Merge top-level strings like "intent_system_prompt" if present
    #    (We assume "intent_system_prompt" is still a single string for now)
    if "intent_system_prompt" in variants and isinstance(variants["intent_system_prompt"], str):
        base_config["AgentConfiguration"]["intent_system_prompt"] = variants["intent_system_prompt"]

    if "human_template" in variants and isinstance(variants["human_template"], str):
        base_config["AgentConfiguration"]["human_template"] = variants["human_template"]

    # 3) Validate base_config
    try:
        validate(instance=base_config, schema=schema)
    except Exception as e:
        sys.stderr.write(f"WARNING: base_config does not conform to schema: {e}\n")

    agent_conf = base_config["AgentConfiguration"]

    # 4) Build combos for top-level "deployment"
    main_deployments = variants.get("deployment", [])
    if isinstance(main_deployments, list) and len(main_deployments) > 0:
        # Filter out inactive
        main_deployments = [d for d in main_deployments if is_active(d)]
        if not main_deployments:
            main_deployments = [agent_conf["deployment"]]
    else:
        main_deployments = [agent_conf["deployment"]]

    # 5) Build combos for "model_parameters" (list -> dict, skip inactive)
    model_param_combinations = build_value_combinations(variants, "model_parameters")

    # 6) For "retrieval"
    retrieval_section = variants.get("retrieval", {})
    if not isinstance(retrieval_section, dict):
        retrieval_section = {}

    # - retrieval deployments, skip inactive
    retrieval_deployments = retrieval_section.get("deployment", [])
    if isinstance(retrieval_deployments, list) and len(retrieval_deployments) > 0:
        retrieval_deployments = [d for d in retrieval_deployments if is_active(d)]
        if not retrieval_deployments:
            retrieval_deployments = [agent_conf["retrieval"]["deployment"]]
    else:
        retrieval_deployments = [agent_conf["retrieval"]["deployment"]]

    # - retrieval parameters
    retrieval_param_combinations = build_value_combinations(retrieval_section, "parameters")

    # ----------------------------------------------------------------------
    ### Handle a list of chat_system_prompt values
    # We now assume `chat_system_prompt` is *always* a list of strings in variants.json
    # e.g. "chat_system_prompt": ["Prompt variant 1", "Prompt variant 2", ...]
    # If it's missing, we fallback to using the base config's single prompt as the only option.
    chat_prompts = variants.get("chat_system_prompt", None)
    if chat_prompts is None or not isinstance(chat_prompts, list) or len(chat_prompts) == 0:
        # fallback to single prompt from base_config
        chat_prompts = [agent_conf["chat_system_prompt"]]
    
    # ----------------------------------------------------------------------

    # 7) Cartesian product
    from itertools import product
    ### Add chat_prompts to the product
    all_combinations = list(
        product(
            main_deployments,
            model_param_combinations,
            retrieval_deployments,
            retrieval_param_combinations,
            chat_prompts  # new dimension
        )
    )
   

    # 8) Limit total variants
    all_combinations = all_combinations[: max_variants]

    # 9) config_version increment
    try:
        version_float = float(agent_conf.get("config_version", 1.0))
    except ValueError:
        version_float = 1.0

    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    valid_count = 0
    # ----------------------------------------------------------------------
    ### Add chat_prompt to the loop 
    for idx, (dep, model_params, ret_dep, ret_params, chat_prompt) in enumerate(all_combinations, start=1):
        new_conf = deepcopy(base_config)
        agent_conf_new = new_conf["AgentConfiguration"]

        # Overwrite
        agent_conf_new["deployment"] = deepcopy(dep)
        agent_conf_new["model_parameters"].update(model_params)
        agent_conf_new["retrieval"]["deployment"] = deepcopy(ret_dep)
        agent_conf_new["retrieval"]["parameters"].update(ret_params)

        ### set the chat_system_prompt from the new dimension
        agent_conf_new["chat_system_prompt"] = chat_prompt

        # Bump config_version
        agent_conf_new["config_version"] = f"{version_float + 0.1 * idx:.1f}"

        # Validate
        try:
            validate(instance=new_conf, schema=schema)
        except Exception as e:
            sys.stderr.write(
                f"Skipping invalid config variant #{idx} due to schema error: {e}\n"
            )
            continue

        # Write
        output_filename = os.path.join(output_dir, f"rag_agent_config_{idx}.yaml")
        with open(output_filename, "w", encoding="utf-8") as out_f:
            out_f.write("# yaml-language-server: $schema=../../../../schemas/agent_config_schema.yaml\n")
            yaml.dump(new_conf, out_f, sort_keys=False)

        print(f"Generated: {output_filename}")
        valid_count += 1

    print(f"Done. Generated {valid_count} YAML files.")

# Run from project root:
# python -m multiagent_evaluation.agents.tools.generate-variants
if __name__ == "__main__":
    print("main")
    args = parse_args()
    generate_variants(
        variants_path=args.variants_json,
        schema_yaml=args.schema_yaml,
        base_config_yaml=args.base_config_yaml,
        max_variants=args.max_variants,
        output_dir=args.output_dir
    )