import os
import json
from typing import List

# initialize all environment variables from .env file
#from opentelemetry import trace

from dotenv import load_dotenv
load_dotenv()
import pandas as pd

from langchain_core.prompts import PromptTemplate
from typing_extensions import Annotated, TypedDict

from multiagent_evaluation.utils.utils import configure_tracing, get_credential, configure_logging, load_agent_configuration
from multiagent_evaluation.aimodel.ai_model import AIModel

# Configure tracing and logging
logger = configure_logging()
tracer = configure_tracing(__file__)

#class for structured output
class Prompt(TypedDict):
    prompt: str
    description: str
    
#class for structured output
class PromptGeneratorOutput (TypedDict):
    prompts: List[Prompt] 


class PromptGenerator:
    def __init__(self, agent_config:dict = None) -> None:

        logger.info("PromptGenerator.Initializing PromptEvaluator")
        with tracer.start_as_current_span("PromptGenerator.__init__span") as span:
            try:
                if agent_config is None:
                    # load configuration from default variant yaml 
                    logger.info("PromptGenerator.__init__#agent_config is empty, loading default configuration")
                    agent_config = load_agent_configuration("agents/prompt_generator", "prompt_generator_config.yaml")
               
                logger.info(f"__init__.agent_config = {agent_config}")
                span.set_attribute("agent_config", agent_config)
                
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
                logger.error(f"PromptEvaluator.__init__#exception= {e}")
                raise e

    def __call__(self, prompt:str, evaluation_dataset:str, evaluation_scores:str) -> str:
        logger.info("PromptGenerator.__call__#PromptGenerator is called")
        return self.generate_prompts(prompt, evaluation_dataset, evaluation_scores)
    
    def __load_prompt__(self, config_path :str, config_file_name, prompt_attribute_name: str) -> str:
        agent_config = load_agent_configuration(config_path, config_file_name)
        return agent_config["AgentConfiguration"][prompt_attribute_name]
        
    def generate_prompts(self, prompt:str, evaluation_dataset:pd.DataFrame) -> dict:
        logger.info("PromptGenerator.generate_prompts#Evaluating prompt")
        with tracer.start_as_current_span("PromptGenerator.generate_prompts_span") as span:
            prompt_template = PromptTemplate.from_template(self.agent_config["AgentConfiguration"]["system_prompt"])
            #prompt = prompt_template.format(prompt=prompt, input=input)
            #logger.info(f"PromptEvaluator.evaluate_prompt#prompt = {prompt}")
            span.set_attribute("prompt", prompt)
            span.set_attribute("evaluation_dataset", evaluation_dataset)
            chain = prompt_template | self.aimodel.llm().with_structured_output(PromptGeneratorOutput) 
            return chain.invoke({"prompt": prompt, "evaluation_dataset": evaluation_dataset.to_json(orient="records")})
             
# To run locally from project root directory: python -m multiagent_evaluation.agents.prompt_generator.prompt_generator   
if __name__ == "__main__":

    prompt_generator = PromptGenerator()
    #load prompt from yaml file
    prompt = prompt_generator.__load_prompt__("agents/rag", "rag_agent_config.yaml", "chat_system_prompt")
    print(f"Prompt = {prompt}")

    evaluation_dataset = """
    [{
        "session_id": 1,
        "question": "What's Microsoft Fabric?",
        "answer": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses various services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. Fabric integrates these components into a cohesive stack, simplifying analytics requirements by offering a seamlessly integrated, user-friendly platform. Key features: Unified data storage with OneLake, AI capabilities embedded within the platform, Centralized data management and governance,SaaS model.",
        "context": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. It offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions.",
        "outputs.output": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. Fabric offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases.\\n\\nFabric operates on a Software as a Service (SaaS) model, providing a seamlessly integrated, user-friendly platform that simplifies analytics requirements. It centralizes data storage with OneLake, integrates various components into a cohesive stack, and embeds AI capabilities, allowing users to transition raw data into actionable insights easily.\\n\\nThe platform unifies components from Power BI, Azure Synapse Analytics, Azure Data Factory, and more into a customized user experience tailored for different roles like data engineers, scientists, or warehousing professionals. It ensures security, governance, and centralized administration across all experiences, allowing creators to focus on their work without needing to manage the underlying infrastructure.",
        "relevance": 5.0,
        "groundedness": 3.0,
        "similarity": 5.0,
        "coherence": 5.0
    },
    {
        "session_id": 2,
        "question": "What is OneLake in Microsoft Fabric?",
        "answer": "OneLake in Microsoft Fabric is a unified storage system built on Azure Data Lake Storage (ADLS) Gen2. It provides a single SaaS experience and a tenant-wide store for data, serving both professional and citizen developers. OneLake simplifies data management by eliminating the need for understanding infrastructure concepts such as resource groups, RBAC, Azure Resource Manager, redundancy, or regions. It ensures easy data discovery, sharing, and uniform enforcement of policy and security settings, eliminating data silos.",
        "context": "Microsoft Fabric integrates separate components into a cohesive stack. Instead of relying on different databases or data warehouses, you can centralize data storage with OneLake. AI capabilities are seamlessly embedded within Fabric, eliminating the need for manual integration. With Fabric, you can easily transition your raw data into actionable insights for business users. OneLake provides the data storage backbone for Fabric, bringing together data from multiple sources.",
        "outputs.output": "OneLake is the data lake that serves as the foundation for all workloads in Microsoft Fabric. It is built into the Fabric platform and provides a unified location to store all organizational data. Here are the key points about OneLake:\\n\\n1. **Unified Storage**: OneLake eliminates data silos by providing a single, unified storage system for all developers, ensuring easy data discovery, sharing, and uniform enforcement of policy and security settings.\\n\\n2. **Built on ADLS Gen2**: OneLake is built on Azure Data Lake Storage (ADLS) Gen2, offering a single SaaS experience and a tenant-wide store for data.\\n\\n3. **Hierarchical Structure**: OneLake is hierarchical, allowing for easy management across the organization. The tenant maps to the root of OneLake, with the ability to create multiple workspaces (folders) within a tenant.\\n\\n4. **Prewired Experiences**: All Microsoft Fabric compute experiences, such as Data Engineering, Data Warehouse, Data Factory, and Power BI, are prewired to use OneLake as their native store without requiring extra configuration.\\n\\n5. **Shortcut Feature**: OneLake allows instant mounting of existing Platform as a Service (PaaS) storage accounts using the Shortcut feature, enabling access to data stored in Azure Data Lake Storage without migration.\\n\\n6. **Single Instance per Tenant**: There is only one OneLake per tenant, providing a single-pane-of-glass file-system namespace that spans across users, regions, and clouds.\\n\\nOverall, OneLake simplifies the data management experience within Microsoft Fabric and facilitates collaboration among developers and business units.",
        "relevance": 5.0,
        "groundedness": 5.0,
        "similarity": 4.0,
        "coherence": 5.0
    },
    {
        "session_id": 3,
        "question": "What are the core services in Microsoft Fabric?",
        "answer": "The core services include Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, Databases and Power BI. These services are designed to work together seamlessly, providing a comprehensive set of analytics experiences tailored to specific user roles and tasks. The platform operates on a Software as a Service (SaaS) model, ensuring simplicity, integration, and centralized management.",
        "context": "Microsoft Fabric offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. These services allow users to handle data end-to-end, from ingestion to analytics, all within a unified platform. The platform's integrated services enable users to transform raw data into actionable insights through various analytics workflows.",
        "outputs.output": "The core services in Microsoft Fabric include:\\n\\n1. **Data Engineering** - This service focuses on preparing and transforming data for analysis.\\n2. **Data Factory** - A service for data integration and orchestration.\\n3. **Data Science** - This service enables data scientists to build and deploy models.\\n4. **Data Warehouse** - A service for storing and analyzing large volumes of structured data.\\n5. **Real-Time Intelligence** - This service provides capabilities for analyzing data in real-time.\\n6. **Industry Solutions** - Tailored solutions for specific industry needs.\\n7. **Power BI** - A business analytics service for visualizing and sharing insights from data.\\n\\nThese services are integrated into a unified environment, tailored for distinct user roles and tasks.",
        "relevance": 4.0,
        "groundedness": 1.0,
        "similarity": 4.0,
        "coherence": 5.0
    },
    {
        "session_id": 4,
        "question": "How does Microsoft Fabric simplify analytics?",
        "answer": "Microsoft Fabric simplifies analytics by providing a unified, end-to-end platform that integrates various data and analytics services into a single, user-friendly environment. The key points: Unified platform, SaaS model,Centralized Data Storage, centralized Governance and Security, easy of use, embedded AI ",
        "context": "With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions. It enables analytics and data handling within a single ecosystem, reducing the complexity associated with managing disparate services.",
        "outputs.output": "Microsoft Fabric simplifies analytics in several ways:\\n\\n1. **Unified Solution**: It is an end-to-end analytics and data platform that integrates various services like Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases into a single, cohesive stack. This eliminates the need for assembling different services from multiple vendors.\\n\\n2. **Centralized Data Storage**: Fabric allows users to centralize data storage with OneLake, rather than relying on different databases or data warehouses. \\n\\n3. **Seamless Integration**: AI capabilities are embedded within Fabric, which removes the need for manual integration and simplifies the transition from raw data to actionable insights.\\n\\n4. **User-Friendly Experience**: The platform provides tailored analytics experiences designed for specific user roles and tasks, making it easier for business owners to access and utilize data intuitively.\\n\\n5. **Governance and Security**: Fabric ensures centralized administration and governance, with automatic permissions and data sensitivity labels applied across all services, which simplifies management and enhances security.\\n\\nOverall, Microsoft Fabric streamlines the analytics process by providing an integrated, user-friendly environment that supports various analytics needs without the complexity of managing multiple tools and services.",
        "relevance": 5.0,
        "groundedness": 5.0,
        "similarity": 5.0,
        "coherence": 5.0
    },
    {
        "session_id": 5,
        "question": "How is AI integrated in Microsoft Fabric?",
        "answer": "AI capabilities are embedded within Fabric, removing the need for separate AI integrations. The keys benefits: Embedded AI Capabilities, Copilot in Microsoft Fabric, Unified Data Management,Automated Machine Learning, Centralized Governance and Security ",
        "context": "AI capabilities are seamlessly embedded within Fabric, eliminating the need for manual integration. This integration simplifies workflows, allowing users to incorporate AI and ML processes directly within their analytics operations. Fabric’s AI capabilities help users transition raw data into actionable insights for business users, supporting smarter decision-making and data-driven processes within enterprises.",
        "outputs.output": "AI capabilities are seamlessly embedded within Microsoft Fabric, eliminating the need for manual integration. This integration allows users to easily transition their raw data into actionable insights for business users. Additionally, Copilot and other generative AI features in preview provide new ways to transform and analyze data, generate insights, and create visualizations and reports within the platform.",
        "relevance": 4.0,
        "groundedness": 3.0,
        "similarity": 4.0,
        "coherence": 5.0
    }]
    """
    df = pd.read_json(evaluation_dataset)
    res = prompt_generator.generate_prompts(prompt, df)
    for p in res["prompts"]:
        print(f"Prompt = {p['prompt']}")