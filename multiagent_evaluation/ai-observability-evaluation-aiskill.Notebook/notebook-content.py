# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

# Welcome to your new notebook
# Type heimport requests
import requests
import json
import pprint
from synapse.ml.mlflow import get_mlflow_env_config



# the URL could change if the workspace is assigned to a different capacity
url = "https://api.fabric.microsoft.com/v1/workspaces/7e98e80c-75fe-4d7c-b854-e8270af71e9cNaN"
configs = get_mlflow_env_config()

headers = {
    "Authorization": f"Bearer {configs.driver_aad_token}",
    "Content-Type": "application/json; charset=utf-8"
}

question = "{userQuestion: \"what's average relevancy for gpt-4o-mini model?\"}"


response = requests.post(url, headers=headers, data = question)


print(f"response : {response.content}")

response = json.loads(response.content)

print(response["executedSQL"])

print("")

print(response["result"])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

help(requests.models.Response)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
