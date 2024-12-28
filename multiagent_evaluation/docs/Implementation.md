## Implementation 

### Tracing 
- To see local traces run from browser : http://127.0.0.1:23337/
- You need to be sure that Prompt Flow service is running on your local machine
- Run from command line: pf service status to check if service is running
- Run from command line: pf service start to start the service
Run from browser: http://127.0.0.1:23337/, 
Ckick on "Show Gannt" button in uppper right corner

Open AI Studio App Insights to see traces

Open Tracing In AI Studio:

https://ai.azure.com/projectflows/trace/run/main_rag_flow_rv7rz322_20241026_111214_744424/details?wsid=/subscriptions/f19e5692-7cfd-4d6a-bea5-8c8ec0c949a9/resourcegroups/genai-workshop-1-rg/providers/Microsoft.MachineLearningServices/workspaces/genai-workshop1-proj&tid=16b3c013-d300-468d-ac64-7eda0820b6d3

### Logging 
TDOO


### Start service locally run:

pf flow serve --source ./ --port 8080 --host localhost
Ask questions with a context:
1. What's Microsoft's Fabric?
2. What's Fabric Data Pipeline?
3. Does it support Cosmos DB as a data source?
4. List the main data sources it does support

### Evaluate the application 
Calculate main metrics: Groundeness, Coherence, Similarity, Relevance,
Prepare Prompty per metric(system prompt)


### Depolyment






This project is a collection of demos that show how to use Azure Data and AI services to build and deploy AI applications.