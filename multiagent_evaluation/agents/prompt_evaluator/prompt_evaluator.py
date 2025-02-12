"""
Module: evaluator
This module contains the Evaluator class which is responsible for evaluating prompts using a language model.  
The Evaluator class uses the AIModel class to interact with the language model and evaluate prompts.
Classes:
    Evaluator: A class to evaluate prompts using a language model.
Usage:
    To run locally from the project root directory: 
    python -m multiagent_evaluation.agents.promptgen.prompt_evaluator.evaluator

    
    A class to evaluate prompts using a language model.
    Methods:
        __init__(agent_config: dict = None) -> None:
            Initializes the Evaluator with the given agent configuration.
        __call__(prompt: str, input: str) -> str:
            Calls the evaluate_prompt method to evaluate the given prompt and input.
        evaluate_prompt(prompt: str, input: str) -> str:
            Evaluates the given prompt and input using the language model.
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from multiagent_evaluation.utils.utils import configure_tracing, configure_logging, load_agent_configuration
from multiagent_evaluation.aimodel.ai_model import AIModel

# initialize all environment variables from .env file
load_dotenv()
# Configure tracing and logging
logger = configure_logging()
tracer = configure_tracing(__file__)

class Evaluator:

    def __init__(self, agent_config:dict = None) -> None:

        logger.info("Evaluator.Initializing")
        
        try:
            if agent_config is None:
                # load configuration from default variant yaml 
                logger.info("Evaluator.__init__#agent_config is empty, loading default configuration")
                agent_config = load_agent_configuration("agents/promptgen/prompt_evaluator", "evaluator_config.yaml")
                
            logger.info(f"__init__.agent_config = {agent_config}")
            
            self.api_key = os.getenv("AZURE_OPENAI_KEY")
            # check if agent config is not None - throw exception
            if agent_config is None or self.api_key is None:
                logger.error("agent config and api_key are required")
                raise ValueError("agent config and api_key are required")

            self.agent_config = agent_config

            # init the AIModel class enveloping a LLM model
            self.aimodel = AIModel(
                azure_deployment=self.agent_config["AgentConfiguration"]["deployment"]["name"],
                openai_api_version=self.agent_config["AgentConfiguration"]["deployment"]["openai_api_version"],
                azure_endpoint=self.agent_config["AgentConfiguration"]["deployment"]["endpoint"],
                api_key=self.api_key,
                model_parameters={"temperature": self.agent_config["AgentConfiguration"]["model_parameters"]["temperature"]}
            )
            
        except Exception as e:
            logger.error(f"Evaluator.__init__#exception= {e}")
            raise e

    def __call__(self, prompt:str, input:str) -> str:
        logger.info("Evaluator.__call__#Evaluator is called")
        return self.evaluate_prompt(prompt, input)
    
    def evaluate_prompt(self, prompt:str, input:str) -> str:
        logger.info("Evaluator.evaluate_prompt#Evaluating prompt")
        prompt_template = PromptTemplate.from_template(self.agent_config["AgentConfiguration"]["system_prompt"])
        #prompt = prompt_template.format(prompt=prompt, input=input)
        #logger.info(f"PromptEvaluator.evaluate_prompt#prompt = {prompt}")
        chain = prompt_template | self.aimodel.llm() | StrOutputParser()
        return chain.invoke({"input": input, "prompt": prompt})
        
    
        
# To run locally from project root directory: python -m multiagent_evaluation.agents.promptgen.prompt_evaluator.evaluator    
if __name__ == "__main__":

    evaluator = Evaluator()
    result = evaluator(prompt="You are a domain-specific technology assistant. When you answer, you must verify that each point you include is explicitly stated in the context. 1.	Read the user’s question. 2.	Check the context for matching information. 3.	Answer solely using details found in the context. 4.	If you cannot find information in the context, politely ask for more details.", \
        input="""{{
    "question": {{
        "0": "What's Microsoft Fabric?",
        "1": "What is OneLake in Microsoft Fabric?",
        "2": "What are the core services in Microsoft Fabric?",
        "3": "How does Microsoft Fabric simplify analytics?",
        "4": "How is AI integrated in Microsoft Fabric?"
    }},
    "answer": {{
        "0": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses various services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. Fabric integrates these components into a cohesive stack, simplifying analytics requirements by offering a seamlessly integrated, user-friendly platform. Key features: Unified data storage with OneLake, AI capabilities embedded within the platform, Centralized data management and governance,SaaS model.",
        "1": "OneLake in Microsoft Fabric is a unified storage system built on Azure Data Lake Storage (ADLS) Gen2. It provides a single SaaS experience and a tenant-wide store for data, serving both professional and citizen developers. OneLake simplifies data management by eliminating the need for understanding infrastructure concepts such as resource groups, RBAC, Azure Resource Manager, redundancy, or regions. It ensures easy data discovery, sharing, and uniform enforcement of policy and security settings, eliminating data silos.",
        "2": "The core services include Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, Databases and Power BI. These services are designed to work together seamlessly, providing a comprehensive set of analytics experiences tailored to specific user roles and tasks. The platform operates on a Software as a Service (SaaS) model, ensuring simplicity, integration, and centralized management.",
        "3": "Microsoft Fabric simplifies analytics by providing a unified, end-to-end platform that integrates various data and analytics services into a single, user-friendly environment. The key points: Unified platform, SaaS model,Centralized Data Storage, centralized Governance and Security, easy of use, embedded AI ",
        "4": "AI capabilities are embedded within Fabric, removing the need for separate AI integrations. The keys benefits: Embedded AI Capabilities, Copilot in Microsoft Fabric, Unified Data Management,Automated Machine Learning, Centralized Governance and Security "
    }},
    "context": {{
        "0": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. It offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions.",
        "1": "Microsoft Fabric integrates separate components into a cohesive stack. Instead of relying on different databases or data warehouses, you can centralize data storage with OneLake. AI capabilities are seamlessly embedded within Fabric, eliminating the need for manual integration. With Fabric, you can easily transition your raw data into actionable insights for business users. OneLake provides the data storage backbone for Fabric, bringing together data from multiple sources.",
        "2": "Microsoft Fabric offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. These services allow users to handle data end-to-end, from ingestion to analytics, all within a unified platform. The platform's integrated services enable users to transform raw data into actionable insights through various analytics workflows.",
        "3": "With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions. It enables analytics and data handling within a single ecosystem, reducing the complexity associated with managing disparate services.",
        "4": "AI capabilities are seamlessly embedded within Fabric, eliminating the need for manual integration. This integration simplifies workflows, allowing users to incorporate AI and ML processes directly within their analytics operations. Fabric\u2019s AI capabilities help users transition raw data into actionable insights for business users, supporting smarter decision-making and data-driven processes within enterprises."
    }},
    "output": {{
        "0": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that need a unified solution. It integrates data movement, processing, ingestion, transformation, real-time event routing, and report building into a single, user-friendly platform. Fabric offers a comprehensive suite of services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. It operates on a Software as a Service (SaaS) model, simplifying analytics requirements by centralizing data storage with OneLake and embedding AI capabilities seamlessly.",
        "1": "OneLake in Microsoft Fabric is the foundational data lake on which all Fabric workloads are built. It provides a unified location for storing all organizational data, eliminating data silos and simplifying data management. Built on ADLS Gen2, OneLake offers a single SaaS experience with a tenant-wide store for data, which serves both professional and citizen developers. It provides easy data discovery, sharing, and uniform policy enforcement without the need for understanding complex infrastructure concepts. OneLake's hierarchical structure allows for easy management, with the ability to create multiple workspaces and lakehouses within a tenant.",
        "2": "The core services in Microsoft Fabric include Data Factory, Data Engineering, Data Warehouse, Data Science, Real-Time Analytics, Power BI, and Data Activator. These services are integrated into a unified SaaS foundation.",
        "3": "Microsoft Fabric simplifies analytics by providing a unified, end-to-end platform that integrates various components needed for data processing and analysis. It eliminates the need to assemble different services from multiple vendors by offering a cohesive suite that includes Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. This integration allows for seamless data movement, processing, ingestion, transformation, and report building. Additionally, Fabric provides a centralized data storage solution with OneLake and embeds AI capabilities, reducing the need for manual integration and enabling easy transition from raw data to actionable insights.",
        "4": "AI is seamlessly embedded within Microsoft Fabric, eliminating the need for manual integration. It is a foundational part of the platform, accelerating the data journey and enabling tasks such as data transformation and insight generation. AI capabilities are integrated across various components like Data Engineering, Data Factory, Data Science, and more, enhancing productivity and simplifying analytics processes."
    }}
}}"""
 )
    
    print( f" Result = {result}")
    print( f" Result type= { type(result)}")
    #d = json.loads(result)
    #print( f" Result dictionary = {d}")