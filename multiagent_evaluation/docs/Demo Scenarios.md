# Demo

1. Clone the repository to your local machine.
    git clone ...

2. Navigate to the `aidemos/llmops` directory.

3. Run the following command to install the dependencies.
    **pip install -r requirements.txt**

## Start chat Session

1. Start the local application server, in commad line run:
    Make sure you are in the `aidemos/llmops` directory.    
    **pf flow serve --source ./ --port 8080 --host localhost**
    

2. The chat UI will be opened automatically in your default browser. If not, you can manually open the browser and navigate to `http://localhost:8080`. 
    
3. Start the chat. Ask some questions about the Microsoft Fabric or Azure AI Studio. Select some session id and use it during the coneversation.
Examples for the chat session are:
    - What is Fabric Data Pipeline?  
        Session ID 1      
    - What're the data source it support? 
        Session ID 1
    - Does it support Cosmos DB as a data source?
        Session ID 1
    - List the main data sources it does support.
        Session ID 1

4. Check the trace
   From command line run
    **pf service status** 
    Notice the prompt flow port number and open the browser and navigate to `http://localhost:<port_number>`. 
    Select the first trace m click on it. 
    Select the "Show Gantt" button in the upper right corner.

5. Explore the traces and logs


## Evaluation

Evaluation metrics are:
- Groundeness
- Coherence
- Similarity
- Relevance

**TODO** - add metrics description and how to calculate them


To run evaluation you must prepare a test data set, this is crucial for the evaluation.
The data set is located in ./rag/data.json file

The evaluation process first runs the entire RAG flow and afterwards run evaluation on the generated responses.

In llmops directory run the following command:
**python -m runflow_local**

The code will be executed locally and you will see the evaluation results in the console and the traces in the browser as described above.

Behind the scenes the logs and traces are send to blob storage and Event Hub for further analysis.
TODO show prompt flow plugin in VS Code



## Statistics in Fabric

Open Fabric Portal and navigate to the the custom dashboard.
TODO - add more details


This demo depicts the developmend life cycle of Gen AP prpject.  


