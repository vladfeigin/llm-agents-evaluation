# Framework for evaluating LLM-based agents

## Intro

When developing LLM-based agents, common frequently asked questions are:

- How to evaluate agents quality?
- How to monitor them?
- How to find the best model parameters and prompts?

This project offers a comprehensive framework, grounded in real-world production experience, for the evaluation and monitoring of LLM Agents.

## Project Description

This project provides the framework of developing and evaluating LLM-based applications. It includes a conversational bot, the RAG Agent, which allows users to ask questions about the content in their documents. The RAG Agent serves solely as an example of an LLM agent being evaluated.
The project showcases effective methods for monitoring and evaluating LLM-based applications, with a strong emphasis on robust evaluation techniques.
In my tests, this approach helped identify the best performing configuration, resulting in an overall improvement of nearly 10% in evaluation metrics.
In addition, the project provides tools to generate multiple agent configurations (variants) and evaluate them to find the best performing one. An agent configuration (variant) is a YAML file that contains all important agent parameters such as prompts, LLM, model parameters, and more.

### Project Structure And Modules

- **multiagent_evaluation**: main project folder.
  - **agents**: LLM agents implementations.
    - **rag**: RAG (Retrieval Augmented Generation) Agent implementation. This agent serves as a conversational bot, allowing users to ask questions about their documents. In this project, we use the RAG Agent to demonstrate the evaluation and monitoring methods.
  - **prompt_generator**: This module generates various prompts for LLM Agent evaluations to find the most performing ones.
  - **orchestrator**: Agent orchestrating the agents' evaluations.
  - **tools**: Utilities for evaluation.
  - **data_ingestion**: Utility for uploading PDF documents and indexing them into Azure AI Search.
  - **aimodel**: Wrapper on top of LLM models.
  - **aisearch**: Wrapper for search functionality. Integrates with Azure AI Search service as a search engine.
  - **kusto_scripts**: Kusto (Azure Data Explorer) scripts for processing traces and logs.
  - **msfabric**: Real-Time LLM evaluation dashboard and Kusto queries for constructing the dashboard.
  - **session_store**: Simple, in-memory session store for storing user sessions, used in the RAG Agent.
  - **utils**: Common utility functions.
  - **docs**: Project documentation.

#### Project Services and LLM Frameworks

- **Azure AI Foundry**: Model deployments, playground, manual and automatic evaluations.
- **Azure AI Search**: Retrieval engine.
- **Azure Document Intelligence**: Documents semantic chunking.
- **Microsoft Fabric**: Observability and evaluation results analysis.
- **Langchain**: Popular LLM framework with easy integrations with Azure AI Search and Azure Document Intelligence.

**Note**: This project goal is demonstrating the framework for developing and evaluating LLM-based applications, it uses RAG Agent just as an example.

## Evaluating LLM-Based Agents

#### Evaluation Datasets

When planning your evaluation strategy, start by preparing **evaluation data sets**.
The data sets emulate your actual interaction with LLM or with LLM based Agent.
This is a key part of any AI-based project and should be tackled in the early stages.

- Domain experts should prepare the data sets.
- Tailor the data sets to your specific use case.
- Start with a small sample of 20-30 flow examples.
- Regularly update data sets with real production examples.
- For a multi-agent system, create a dedicated data set for each agent.

For example the evaluation data set for conversational flow (chat), could have the following schema:

**`question, answer, context, chat-history`**

`question` is the question asked by the user.

`chat-history` is a list of previous questions and answers.

`answer` is the expected answer.

`context` is the background information related to the question, serving as the ground-truth for the model to generate an accurate answer.

Test data sets are typically constructed as JSONL files, with each line being an independent JSON describing one conversation turn.

Example:

```json
{
    "session_id": "1",
    "question": "What's Microsoft Fabric?",
    "answer": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses various services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. Fabric integrates these components into a cohesive stack, simplifying analytics requirements by offering a seamlessly integrated, user-friendly platform. Key features: Unified data storage with OneLake, AI capabilities embedded within the platform, Centralized data management and governance, SaaS model.",
    "context": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. It offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions."
}
```

#### Evaluation Metrics

In this project, we use the following evaluation metrics:

##### Evaluation Metrics for Conversational Agents

- **Groundedness**  
    Evaluates whether the generated response is consistent and accurate with respect to the provided context in a RAG question-and-answer scenario.

- **Relevance**  
    Measures how effectively the response addresses the query by evaluating its accuracy, completeness, and direct connection to the question.

- **Coherence**  
    Assesses the logical and orderly presentation of ideas, ensuring the response is easy to follow and understand.

- **Similarity**  
    Determines the level of resemblance between the generated text and its ground truth in relation to the query.

In this project we compute these metrics on a scale from 1 to 5. 
For this project, we use the Azure Evaluation SDK, which offers various built-in evaluation metrics. 
For more details, please refer to the [documentation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk).


##### Metrics Selection

When selecting evaluation metrics, consider the following:

- **Determine Relevant Metrics**  
    Decide which metrics to measure based on your use case.

- **Weight Significant Metrics**  
    Assign greater weights to more significant metrics according to your workflow, and calculate an aggregated score.

- **Multi-Agent Considerations**  
    In a multi-agent system, each agent may utilize a distinct set of evaluation metrics.

- **Use Case Examples**  
    - For conversational flows: consider metrics such as **relevancy**, **similarity**, and **groundedness**.  
    - For summarization tasks: the **similarity** metric might be most relevant.

For this project, we use the [Azure Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk), which includes a variety of built-in evaluation metrics.

#### Manual Evaluation

The evaluation dataset and metrics enable manual checks to validate the LLM, model parameters, and prompts, ensuring your idea works. This is essential step in developing LLM-based applications. 

#### Automatic Evaluation Implementation

Add automatic evaluation to your project by using an advanced LLM model _as a judge_ for AI-assisted evaluation. 
It's recommended to use state-of-the-art LLMs for this purpose. 
The project in this repo uses the [Azure Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk) and [Azure AI Foundry Evaluator Library](http://ai.azure.com/), which offers ready prompts specifically designed for evaluations.


#### Integration with CI/CD Pipeline

Incorporate automatic evaluations into your CI/CD pipeline. 
Trigger automatic evaluations on any changes to an Agent and fail the build if metrics fall below set thresholds.

#### Agent Configuration

The core of the evaluation and monitoring process is the concept of **_Agent Configuration_** or **_Variant_**. For each Agent, a configuration file in YAML format is created, encompassing all essential settings such as prompts, LLM, model parameters, agent name, application name and more. Any update to this configuration file results in a new revision. During the evaluation of the Agents, or during their ongoing usage, the current Agent configuration revision is recorded in the logs and traces. This practice enables the comparison of evaluation metrics for each Agent for each configuration revision, facilitating the assessment of how specific configuration revisions impact their performance. It is important to note that logs and traces are collected and analyzed within Microsoft Fabric, however you can use other services for this purpose.

Example of configuration file:

```yaml
AgentConfiguration:
    description: "RAG Agent configuration"
    config_version: 1.1
    application_version: "1.0"   
    application_name: "rag_llmops_workshop" 
  
    agent_name: "rag_agent"
    model_name: "gpt-4o-mini"
    model_version: "2024-08-01"

    openai_api_version: "2024-10-01-preview"

    model_deployment: "gpt-4o-mini"
    model_deployment_endpoint: "https://openai-australia-east.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-01-preview"
  
    retrieval: 
            search_type: "hybrid"
            top_k: 5
            embedding_endpoint: "https://openai-australia-east.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
            embedding_deployment: "text-embedding-ada-002"
            index_name: "micirosoft-tech-stack-index-0"
            index_semantic_configuration_name: "vector-llmops-workshop-index-semantic-configuration"
    model_parameters:
        temperature: 0.0
        seed: 42
    intent_system_prompt: > 
                Your task is to extract the user’s intent by reformulating their latest question into a standalone query that is understandable without prior chat history. 
                Analyze the conversation to identify any contextual references in the latest question, and incorporate necessary details to make it self-contained. 
                Ensure the reformulated question preserves the original intent. Do not provide an answer; only return the reformulated question. 
                For example, if the latest question is ‘What about its pricing?’ and the chat history discusses Microsoft Azure, reformulate it to ‘What is the pricing of Microsoft Azure?’

    chat_system_prompt: > 
                You are a knowledgeable assistant specializing only in technology domain. 
                Deliver concise and clear answers, emphasizing the main points of the user’s query.
                Your responses should be based exclusively on the context provided in the prompt; do not incorporate external knowledge.
                If the provided context is insufficient to answer the question, request additional information.
  
                <context>
                {context}
                </context>"
```

#### Running Evaluations

Run evalations during development and in CI/CD.
For each LLM Agent implement a specific evaluation script.
For example, in this project the evaluation script for the RAG Agent is  `./multiagent_evaluation/agents/rag/evaluation/evaluation_implementation.py`.
This script uses the Azure Evaluation SDK to calculate the conversational evaluation metrics: Groundedness, Relevance, Similarity and Coherence.
The script `./multiagent_evaluation/agents/tools/evaluate.py` is a generic script which runs specific LLM Agent evaluation implementation on evaluation data sets.

For example, to run the evaluation for the RAG Agent, execute the following command from the project root folder:

```sh
    python -m multiagent_evaluation.agents.tools.evaluate \
    --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
    --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
    --config_dir agents/rag \
    --config_file rag_agent_config.yaml \
    --eval_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
    --dump_output
    --mode single
```

where the parameters are:

- `agent_class`: LLM Agent main implementation class.
- `eval_fn`: Evaluation function implementation. This function calculates the evaluation metrics.
- `config_dir`: Directory where the LLM Agent configuration file is located (more details in the next sections).
- `config_file`: LLM Agent configuration file name.
- `eval_dataset`: Evaluation data set.
- `dump_output`: Flag to dump the evaluation results to the output files.
- `mode`: Evaluation mode (`single` or `multiple`). In the single mode, the evaluation is performed for a single LLM Agent configuration. In the multiple mode, the evaluation is performed for multiple configurations.

Running this command will first evaluate the agent on the provided evaluation data set, calculate the evaluation metrics for the evaluated agent (RAG Agent in this example), and output the results to the `batch_eval_results_timestamp.json` and `eval_results_timestamp.json` files in the project root folder, as well as to the console.

Here is example of evaluation results file (eval_results_timestamp.json):

```json
{
    "session_id": 1,
    "question": "What's Microsoft Fabric?",
    "answer": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses various services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. Fabric integrates these components into a cohesive stack, simplifying analytics requirements by offering a seamlessly integrated, user-friendly platform. Key features: Unified data storage with OneLake, AI capabilities embedded within the platform, Centralized data management and governance,SaaS model.",
    "context": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. It offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions.",
    "outputs.output": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises requiring a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. Fabric offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases.\n\nIt operates on a Software as a Service (SaaS) model, providing a seamlessly integrated, user-friendly platform that simplifies analytics requirements. Microsoft Fabric integrates separate components into a cohesive stack, centralizing data storage with OneLake, and includes embedded AI capabilities to eliminate manual integration. This allows users to transition raw data into actionable insights easily.\n\nFabric combines both new and existing components from Power BI, Azure Synapse Analytics, Azure Data Factory, and more into a unified environment tailored for customized user experiences. It integrates workloads such as Data Engineering, Data Factory, Data Science, Data Warehouse, Real-Time Intelligence, and Power BI into a shared SaaS foundation, ensuring unified management, governance, and discovery across all services.",
    "relevance": 5.0,
    "groundedness": 3.0,
    "similarity": 5.0,
    "coherence": 4.0
}
{
    "session_id": 2,
    "question": "What is OneLake in Microsoft Fabric?",
    "answer": "OneLake in Microsoft Fabric is a unified storage system built on Azure Data Lake Storage (ADLS) Gen2. It provides a single SaaS experience and a tenant-wide store for data, serving both professional and citizen developers. OneLake simplifies data management by eliminating the need for understanding infrastructure concepts such as resource groups, RBAC, Azure Resource Manager, redundancy, or regions. It ensures easy data discovery, sharing, and uniform enforcement of policy and security settings, eliminating data silos.",
    "context": "Microsoft Fabric integrates separate components into a cohesive stack. Instead of relying on different databases or data warehouses, you can centralize data storage with OneLake. AI capabilities are seamlessly embedded within Fabric, eliminating the need for manual integration. With Fabric, you can easily transition your raw data into actionable insights for business users. OneLake provides the data storage backbone for Fabric, bringing together data from multiple sources.",
    "outputs.output": "OneLake is the data lake that serves as the foundation for all workloads within Microsoft Fabric. It is built into the Fabric platform and provides a unified location to store all organizational data, simplifying the management of data across the organization. Here are the key points about OneLake:\n\n1. **Built on ADLS Gen2**: OneLake is built on Azure Data Lake Storage (ADLS) Gen2, providing a single Software as a Service (SaaS) experience.\n\n2. **Single Tenant**: There is only one OneLake per tenant, which offers a single-pane-of-glass file-system namespace that spans across users, regions, and clouds.\n\n3. **Hierarchy and Management**: OneLake is hierarchical, allowing you to create multiple workspaces (akin to folders) within a tenant, and each workspace can contain multiple lakehouses (collections of files, folders, and tables).\n\n4. **Data Centralization**: It eliminates data silos created by individual developers, providing a unified storage system that ensures easy data discovery, sharing, and uniform enforcement of policy and security settings.\n\n5. **Integration with Microsoft Fabric**: All Microsoft Fabric compute experiences are prewired to use OneLake as their native store, which means no extra configuration is needed.\n\n6. **Shortcuts Feature**: OneLake allows for instant mounting of existing Platform as a Service (PaaS) storage accounts and facilitates data sharing without the need to move or duplicate information.\n\nOverall, OneLake simplifies the analytics and data management processes within Microsoft Fabric by providing a centralized and integrated storage solution.",
    "relevance": 5.0,
    "groundedness": 3.0,
    "similarity": 5.0,
    "coherence": 4.0
    ....
}
```

*outputs.output* field is the generated answer by the LLM Agent followed by the evaluation metrics.
Note that outputs.output and evaluation metrics are added on top of the original evaluation data set.

In addition to the evaluation results, the script outputs the aggregated evaluation metrics for the evaluated agent in a file named `batch_eval_output_timestamp.json` in the project root folder.
Here is an example of the aggregated evaluation metrics file:

```json
{
    "metric": "relevance",
    "score": 5.0
}
{
    "metric": "groundedness",
    "score": 3.2
}
{
    "metric": "similarity",
    "score": 4.4
}
{
    "metric": "coherence",
    "score": 4.6
}
```

#### Multiple Configurations Evaluations

You can evaluate many agent configurations (variants) at once by running the evaluation script with the `mode` parameter set to `multiple`.
This allows you to compare the evaluation metrics for each configuration and determine how specific configuration affects the agent performance and select the best performing one.
The agent configuration is stored in YAML file and contains all important agent parameters such as prompts, LLM, model parameters, and more.
For example RAG agent configuration is stored in `./multiagent_evaluation/agents/rag/rag_agent_config_example.yaml` file.
When we instantiate an agent, we pass the configuration file to the agent constructor, this way at any given time we know which configuration is used for the agent. Moreover, we log the configuration revision in the logs and traces for each evaluation. This allows us to compare the evaluation metrics for each agent and determine how specific configuration revisions affect their performance.

Example of ruuning the evaluation for multiple configurations:

```sh
    python -m multiagent_evaluation.agents.tools.evaluate \
  --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
  --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
  --config_dir agents/rag/evaluation/configurations/generated \
  --eval_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
  --mode multiple
```

In `--mode` set to `multiple`, the `--config_dir` parameter specifies the directory where the multiple agent configurations are stored.

#### Generating Multiple Agent Configurations

You can generate multiple agent configurations (variants) and evaluate them to find the best performing one by running the evaluation orchestrator: `.\multiagent_evaluation\agents\orchestrator\evaluation_orchestrator`
This script generates multiple agent configurations based on the base variant configuration file and evaluates them.

For example for RAG agent you can run the following command:

```sh
    python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator \
    --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
    --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
    --agent_config_file_dir agents/rag \
    --agent_config_file_name rag_agent_config.yaml \
    --evaluation_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
    --base_variant ./multiagent_evaluation/agents/rag/variants.json \
    --output_dir ./agents/rag/evaluation/configurations/generated
```

where the parameters are:

- `--agent_class`: LLM Agent main implementation class.
- `--eval_fn`: Evaluation function implementation. This function calculates the evaluation metrics for the agent.
- `--agent_config_file_dir`: Directory where the LLM Agent configuration file is located.
- `--agent_config_file_name`: LLM Agent configuration file.
- `--evaluation_dataset`: Evaluation data set.
- `--base_variant`: Base variant configuration file. This file contains the main configuration parameters which are used to generate multiple configurations. It contains the rules on how to generate the parameters in configurations. This file is used by `./multiagent_evaluation/agents/tools/generate_variants.py` script to generate multiple variants.
- `--output_dir`: Directory where the generated configurations are stored.

Here is example of base variant configuration file:

```json
{
   
        "deployment": [
            {
                "name": "gpt-4o-mini",
                "model_name": "gpt-4o-mini",
                "model_version": "2024-08-01",
                "endpoint": "https://<aoai-instance-name>.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-01-preview",
                "openai_api_version": "2024-10-01-preview"
            },
            {
                "name": "gpt-4o-3",
                "model_name": "gpt-4o",
                "model_version": "2024-08-01",
                "endpoint": "https://<aoai-instance-name>.openai.azure.com/openai/deployments/gpt-4o-3/chat/completions?api-version=2024-10-01-preview",
                "openai_api_version": "2024-10-01-preview"
            },
            {
                "name": "gpt-4o-2",
                "model_name": "gpt-4o",
                "model_version": "2024-05-13",
                "endpoint": "https://<aoai-instance-name>.openai.azure.com/openai/deployments/gpt-4o-2/chat/completions?api-version=2024-10-01-preview",
                "openai_api_version": "2024-10-01-preview"
            },
            {
                "name": "gpt-35-turbo-0301",
                "model_name": "gpt-35-turbo",
                "model_version": "0301",
                "endpoint": "https://<aoai-instance-name>.openai.azure.com/openai/deployments/gpt-35-turbo-0301/chat/completions?api-version=2024-10-01-preview",
                "openai_api_version": "2024-10-01-preview"
            }
        ],
        "model_parameters": [
            {
                "name": "temperature",
                "range": [0.0, 1.0],
                "step": 0.5,
                "default": 0.0,
                "active": "true"
            },
            {
                "name": "top_p",
                "range": [0.1, 0.9],
                "step": 0.5,
                "default": 0.9,
                "active": false
            },
            {
                "name": "frequency_penalty",
                "range": [0.0, 1.0],
                "step": 0.5,
                "default": 0.0,
                "active": false
            },
            {
                "name": "presence_penalty",
                "range": [0.0, 1.0],
                "step": 0.5,
                "default": 0.0,
                "active": false
            }
        ],
        "retrieval": {
            "parameters": [
                {
                    "name": "search_type",
                    "set": ["hybrid", "similarity"],
                    "default": "hybrid"
                },
                {
                    "name": "top_k",
                    "range": [3, 5],
                    "step": 2,
                    "default": 5
                }
            ],
            "deployment": [
                {
                    "model_name": "text-embedding-ada-002",
                    "model_version": "2024-08-01",
                    "openai_api_version": "2024-10-01-preview",
                    "name": "text-embedding-ada-002",
                    "endpoint": "https://<aoai-instance-name>.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
                }
            ]
        },
        "intent_system_prompt": "Your task is to extract the user’s intent by reformulating their latest question into a standalone query that is understandable without prior chat history. Analyze the conversation to identify any contextual references in the latest question, and incorporate necessary details to make it self-contained. Ensure the reformulated question preserves the original intent. Do not provide an answer; only return the reformulated question. For example, if the latest question is ‘What about its pricing?’ and the chat history discusses Microsoft Azure, reformulate it to ‘What is the pricing of Microsoft Azure?’",
        "chat_system_prompt": "You are a knowledgeable assistant specializing only in technology domain. Deliver concise and clear answers, emphasizing the main points of the user’s query. Your responses should be based exclusively on the context provided in the prompt; do not incorporate external knowledge. If the provided context is insufficient to answer the question, request additional information.\n\n<context>\n{context}\n</context>",
        "human_template": "question: {input}"
}
```

The **"Orchestrator"** agent for generating multiple variants performs the following steps:

1. Runs the original variant evaluation to establish baseline metrics.
2. Creates multiple prompt variants based on the initial evaluation results.
3. Evaluates the multiple variants.
4. Selects the best performing variant based on the evaluation results.


![Multiple variants evaluation](multiagent_evaluation/docs/img/multiple-variants-evaluation.png)


After running the evaluation orchestrator, you will have evaluation results for each configuration file.
For example, here you can see top 5 most performing configurations after running the evaluation orchestrator:

![Multiple Agent Configuration Evaluation](multiagent_evaluation/docs/img/Multiple-variants-evaluation-dashboard.png)

#### Monitoring

Monitoring is a crucial part of any LLM-based project. It should be designed and implemented from project beginning, rather than postponed to the final phases.
In this project, we collect data from the RAG Agent, including tokens consumption, model parameters, and evaluation metrics. This data is crucial for cost calculations and quality analysis of LLM-based applications, helping determine the impact of changes like prompts, model parameters, model versions on application quality.
In this project, we use the Open Telemetry standard to collect application traces, logs and metrics.
For simplicity, logs, traces, and metrics are currently sent to Azure Application Insights. From Application Insights, they are forwarded via Azure Event Hub to Microsoft Fabric EventHouse (Kusto). We utilize Kusto Update Policies to flatten the traces and store them in the final tables. Notably, Application Insights supports the Open Telemetry standard, allowing you to view traces and logs in "Transaction Search" during development, which is convenient.

To send the logs and traces to Azure Application Insights, set the `APPLICATIONINSIGHTS_CONNECTION_STRING` environment variable in the `.env` file to your Application Insights connection string.
For visualizing traces in Azure AI Foundry, set the environment variable `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` to `true`.
For more details, refer to the [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/visualize-traces).

Traces, logs and metrics are collected in Microsoft Fabric for further analysis and reporting.
In the next version, we plan to introduce support for Azure Data Explorer for storing and analyzing traces, logs, and metrics.

#### Local Development

To facilitate local development, we utilize Visual Studio Code. The Prompt Flow extension in VS Code simplifies several aspects of local development. For instance, it offers tools to run RAG locally providing a simple web interface for chat.
Havind said that, Prompt Flow is not mandatory for this project.

#### Architecture

![Architecture](multiagent_evaluation/docs/img/architecture.png)

#### Prerequisites

- **Azure AI Foundry project**
- **Azure AI Search service instance**
- **Azure Document Intelligence service instance**
- **Microsoft Fabric** (Trial could be used)
- **Visual Studio Code**
- **Python 3.11** (The project has been tested with Python version 3.11 on Mac)

#### Project Installation

1. Create a project folder and navigate to it:

   ```sh
   mkdir llm-agents-evaluation
   cd llm-agents-evaluation
   ```
2. Clone the project from GitHub:

   ```sh
   git clone https://github.com/vladfeigin/llm-agents-evaluation.git
   ```
3. Create a virtual environment in the root project folder:

   ```sh
   python3.11 -m venv .venv
   ```
4. Activate the virtual environment:

   ```sh
   source .venv/bin/activate
   ```
5. Install dependencies:

   ```sh
   pip install -r ./requirements.txt
   ```
6. Create a `.env` file from the `.env_template` file located in the project root folder:

   ```sh
   cp .env_template .env
   ```
7. Populate the `.env` file with values pertinent to your environment.

#### Ingest your documents to search index

In this step, you will ingest your PDF documents to a new Azure AI Search index.

1. Copy the PDF files into a local folder.
2. Change the directory to the project’s folder (`llm-agents-evaluation`).
3. Run:
   ```sh
   python -m multiagent_evaluation.data_ingestion.semantic_chunking_di --index_name <my index name> --input_folder <my input folder>
   ```

After ingestion, open Azure AI Search in Azure portal and verify the new index is created correctly and contains your documents.

#### Running RAG Agent locally

1. Change the directory to the project root folder.
2. In `./multiagent_evaluation/agents/rag/rag_agent_config.yaml`, change the search index name to the newly created index name from the previous step.
3. From the command line, run:

   ```sh
   pf flow serve --source ./multiagent_evaluation/agents/rag/ --port 8080 --host localhost
   ```

   This will open a chat web interface. Now you can ask questions about the ingested documents.

 ![PF_WebUI](multiagent_evaluation/docs/img/rag_evaluation_web_chat_ui.png)

4. Open Microsoft Fabric Real-Time dashboard. Select Ongoing Page and see the details of using RAG agent.

   ![Fabric_Ongoing](multiagent_evaluation/docs/img/Fabric_ongoing.png)

#### Running RAG Agent evaluation

1. Replace the `./rag/data.json` file with your relevant dataset. This dataset is for RAG Agent evaluation.
2. Once you have created a relevant dataset, **from the project root folder**, execute from the command line:

   ```sh
    python -m multiagent_evaluation.agents.tools.evaluate \
    --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
    --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
    --config_dir agents/rag \
    --config_file rag_agent_config.yaml \
    --eval_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
    --dump_output
    --mode single
   ```

If you have your own LLM Agent implementation, replace the `--agent_class` parameter with your LLM Agent main implementation class.
Replace the `--eval_fn` parameter with your evaluation function implementation.
Change other parameters according to your LLM Agent implementation.

Open Microsoft Fabric Real-Time dashboard. Select Evaluation Page and see the evaluation results.

![Fabric_Evaluation](multiagent_evaluation/docs/img/Fabric_evaluation.png)

Change the model parameters or/and prompts in the `./rag/rag_agent_config.yaml` file and run the evaluation again.

Compare the evaluation metrics with the previous evaluation results:

![Fabric_Evaluation_2](multiagent_evaluation/docs/img/Fabric_evaluation_2.png)

You also have a detailed view of the evaluation results, which can help investigate the differences between evaluations and provide insights for each evaluation test:

![Fabric_evaluation_detailed_1](multiagent_evaluation/docs/img/Fabric_evaluation_detailed_1.png)

![Fabric_evaluation_detailed_2](multiagent_evaluation/docs/img/Fabric_evaluation_detailed_2.png)

3. Run multiple configurations evaluation to find best performing configuration with evaluation orchestrator:

   ```sh
    python -m multiagent_evaluation.agents.orchestrator.evaluation_orchestrator \
    --agent_class multiagent_evaluation.agents.rag.rag_main.RAG \
    --eval_fn multiagent_evaluation.agents.rag.evaluation.evaluation_implementation.eval_batch \
    --agent_config_file_dir agents/rag \
    --agent_config_file_name rag_agent_config.yaml \
    --evaluation_dataset ./multiagent_evaluation/agents/rag/evaluation/data.jsonl \
    --base_variant ./multiagent_evaluation/agents/rag/variants.json \
    --output_dir ./agents/rag/evaluation/configurations/generated
   ```

   Change parameters according to your LLM Agent implementation.

### Observability with Microsoft Fabric

![Fabric Observability Document](multiagent_evaluation/docs/Fabric%20Observability.md)
