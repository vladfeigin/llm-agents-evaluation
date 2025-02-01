#this module is responsible for orchestrating the evaluation of the agents
#it first calls run_and_eval_flow which runs the agent and evaluates the output
#user add the task in natural language and the agent will be evaluated based on the task
#python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator

from multiagent_evaluation.agents.tools.evaluate import run_and_eval_flow
from multiagent_evaluation.agents.prompt_generator.prompt_generator import PromptGenerator
from multiagent_evaluation.utils.utils import load_agent_configuration

#run initial evaluation        
eval_res = run_and_eval_flow("agents/rag", "rag_agent_config.yaml", "./multiagent_evaluation/agents/rag/evaluation/data.jsonl" ,dump_output=False )
#take the evaluation results and create multiple prompts variants based on the initial evaluation

pgen = PromptGenerator(load_agent_configuration("agents/prompt_generator", "prompt_generator_config.yaml" ))

evaluated_agent_config = load_agent_configuration("agents/rag", "rag_agent_config.yaml")
prompts = pgen.generate_prompts(evaluated_agent_config["AgentConfiguration"]["chat_system_prompt"],  eval_res)



if __name__ == "__main__":
    #print(eval_res)
    print(prompts)
    print("Evaluation completed successfully.")