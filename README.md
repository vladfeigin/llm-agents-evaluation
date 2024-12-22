# Methodology of developing and evaluation LLM-based applications
###  
## Intro 

When developing LLM-based applications, common questions include:

How to develop LLM-based application?
How to evaluate their quality?
How to monitor them?

This project provides best practices and real examples based on production experience for the aforementioned questions.

## Methodology steps

#### Evaluation Data Sets 

When planning your evaluation strategy, start by preparing evaluation data sets. 
The data sets emulate your actual interaction with LLM or with LLM based Agent.

Keep these points in mind:
Domain experts should prepare the data sets.
Tailor the data set structure to your use case.
Start with small size to 20-30 samples of your flow.
Regularly update data sets with real production examples.
For a multi-agent system, create a dedicated dataset for each agent.

For example the evaluation data set for conversational flow, could have the following schema:

question, ground-truth, context, chat-history

Example:
{"session_id":"1", "question": "What's Microsoft Fabric?","ground-truth": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses various services such as Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. Fabric integrates these components into a cohesive stack, simplifying analytics requirements by offering a seamlessly integrated, user-friendly platform. Key features: Unified data storage with OneLake, AI capabilities embedded within the platform, Centralized data management and governance,SaaS model.","context": "Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. It encompasses data movement, processing, ingestion, transformation, real-time event routing, and report building. It offers a comprehensive suite of services including Data Engineering, Data Factory, Data Science, Real-Time Analytics, Data Warehouse, and Databases. With Fabric, you don't need to assemble different services from multiple vendors. Instead, it offers a seamlessly integrated, user-friendly platform that simplifies your analytics requirements. Operating on a Software as a Service (SaaS) model, Fabric brings simplicity and integration to your solutions."}


#### Manual evaluation

With an evaluation dataset, you can perform manual evaluations to validate the LLM, model parameters, and prompts. This process helps ensure your idea works and is crucial step in developing LLM-based applications. Azure AI Foundry offers a Manual Evaluation tool. 

#### Metrics 

Decide about metrics you want to measure.

For conversational flows, consider using metrics such as relevancy, similarity, and groundedness.
For summarization tasks it could be similarity metric.
            The relevant metrics for a measurement depend on your use case.
             Assign greater weights to more significant metrics according to your workflow. 
Calculate an aggregated score using these weights.
For instance, if standard phrasing is crucial in a model output, give higher weight to the Similarity metric comparing ground truth and model answers.
In a multi-agent system, each agent may utilize a distinct set of evaluation metrics.
In this project, we utilize the Azure Evaluation SDK, which includes a variety of built-in evaluation metrics.

#### 
#### Automatic Evaluation implementation

Incorporate automatic evaluation into your project.
The implementation depends on your flow. For conversational flows, another advanced model is typically needed for evaluation, the judge model.
The project in this repo uses the Azure Evaluation SDK  and Azure AI Foundry Evaluator Library, which offers ready prompts specifically designed for evaluations.
Integrate automatic evaluations into your CI/CD pipeline. Fail the build if metrics drop below predefined quality thresholds.

#### Running evaluations

Run evalations during development and in CI/CD.
In this project, we use the Azure AI Foundry Prompt Flow SDK to run automatic evaluations on the evaluation data set.


#### Monitoring 

Monitoring is a crucial part of any LLM-based project. It should be designed and implemented from the beginning, rather than postponed to the final phases. 
In this project, we collect data from the RAG Agent, including tokens, model details, and evaluation metrics. This data is crucial for cost calculations and quality analysis of LLM-based applications, helping determine the impact of changes like prompts, model parameters, or models on application quality.
In this project, we use the promptflow-tracing package to collect application traces. 
Prompt Flow tracing follows the Open Telemetry Standard.


#### Local development

To facilitate local development, we utilize Visual Studio Code. The Prompt Flow extension in VS Code simplifies several aspects of local development. For instance, it offers tools to run RAG locally and collects Open Telemetry traces on a local level. 
It is also worth mentioning that we use the Prompt Flow Flex flavor, which provides comprehensive development flexibility.

#### Agent Configuration

For each Agent, we create a configuration file in YAML format, which includes all mandatory settings such as prompts, LLM, model parameters, and more. Any modification to this configuration file results in a new revision. During the evaluation of the Agents, we record the current revisions in the logs. 
Logs are collected and analyzed in Microsoft Fabric.
This allows us to compare the evaluation metrics for each Agent and determine how specific configuration revisions affect their performance. 


## 
## Project description
The primary objective of the project is to illustrate the methods for monitoring and evaluating LLM-based applications. It showcases the functionality of Retrieval Augmented Generation (RAG), placing emphasis on monitoring and evaluation while also addressing certain aspects of local development.


The project comprises the following modules:
aimodel: A wrapper on top of LLM models.
aisearch: A wrapper for search functionality utilizing the Azure AI Search service.
evaluation: This module calculates conversation metrics such as Groundedness, Relevance, Similarity, and Coherence using the Azure Evaluation SDK.
rag: Implements the RAG agent, the project's core component. This is conversational bot, allowing you to ask questions on top of your documents.
data_ingestion: A utility that allows uploading PDF documents and indexing them into Azure AI Search. The RAG Agent uses AI Search to retrieve relevant context for each user question. This model uses Azure Document Intelligence service to semantically chunk the PDF documents and convert the chunks into Markdown language for indexing in Azure AI Search. 
msfabric: This module encompasses a Real-Time LLM evaluation dashboard and KQL (Kusto Query Language) queries utilized for constructing the dashboard.
docs: Project documentation.



#### Project Services and LLM Frameworks 

In this project, we use the following services and LLM frameworks: 
Azure AI Foundry: Model deployments, playground, and manual evaluations.
Azure AI Search: Retrieval engine.
Azure Document Intelligence: Documents semantic chunking.
Microsoft Fabric: Observability and evaluation results analysis.
Langchain: LLM framework with easy integrations with Azure AI Search and Azure Document Intelligence.




#### Architecture
TODO

#### Prerequisites
Azure AI Foundry project.
Azure AI Search service instance.
Azure Document Intelligence service instance.
Microsoft Fabric. Trial could be used. 
Vusual Studio Code
Python 3.11. The project has been tested with Python version 3.11 on Mac

#### Project installation
Clone the project from github.
Create a project folder and run: 
https://github.com/vladfeigin/llm-agents-evaluation.git

In the root project folder create virtual environmewnt: 
python3.11 -m venv .venv

Activate virtual environment: 
   source .venv/bin/activate

Install dependencies: 
   pip install -r ./requirements.txt

Create .env file from .env_template file, located in the project root folder. 
Populate it with values pertinent to your environment.

#### Ingest your documents to search index

In this step, you will ingest your PDF documents to a new Azure AI Search index.
Copy the PDF files into a local folder.
Change the directory to the project’s data_ingestion folder.
Run: python semantic_chunking_di.py --index_name <index name> --input_folder <folder name with the documents >

After ingestion, open Azure AI Search and verify the new index is created correctly and contains your documents.

#### Running RAG Agent locally

Change the directory to the project root folder: rag_evaluation folder.
In ./rag/rag_agent_config.yaml, change the search index name to the newly created index name from the previous step.
From the command line, run: `pf flow serve --source ./ --port 8080 --host localhost`.
This will open a chat web interface.  Now you can ask questions about the ingested documents.
Open Microsoft Fabric Real-Time dashboard. Select Ongoing Page and see the details of using RAG agent. This


#### Running  RAG Agent evaluation

Replace the ./rag/data.json file with your relevant dataset.
This dataset is for RAG Agent evaluation.
Once you have created a relevant dataset, execute the command from the command line:
python -m ./rag_evaluation/runflow_local


#### Fabric

































