import requests
import json
import pprint
#from synapse.ml.mlflow import get_mlflow_env_config


from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
token = credential.get_token("https://management.azure.com/.default")
print(f"Access Token: {token.token}")


# the URL could change if the workspace is assigned to a different capacity
url = "https://msitapi.fabric.microsoft.com/v1/workspaces/64716e77-1595-4e06-9424-58d6b8d03b77/aiskills/881ceb3b-367d-495e-a08d-2dba6100f5b6/query/deployment"


headers = {
    "Authorization": f"Bearer {token.token}",
    "Content-Type": "application/json; charset=utf-8"
}

question = "{userQuestion: \"what's average latency for model gpt-4o-mini  ?\"}"


response = requests.post(url, headers=headers, data = question)

print("")

response = json.loads(response.content)

print(response)
#print(response["executedSQL"])

#print("")

#print(response["result"])