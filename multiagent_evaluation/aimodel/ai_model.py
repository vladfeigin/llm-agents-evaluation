
"""
A wrapper class for the Azure OpenAI model using the Langchain framework.
This class initializes and configures an AzureChatOpenAI model instance with the provided parameters.
It also sets up logging and tracing for monitoring the model's initialization process.
Attributes:
    _llm (AzureChatOpenAI): An instance of the AzureChatOpenAI model.
Methods:
    __init__(azure_deployment, openai_api_version, azure_endpoint, api_key, model_parameters):
        Initializes the AIModel with the specified parameters.
    llm():
        Returns the AzureChatOpenAI model instance.
"""
import os
from logging import getLogger
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from multiagent_evaluation.utils.utils import configure_tracing

# Load environment variables from .env file
load_dotenv()

# Configure logging
# Logging calls with this logger will be tracked
logger = getLogger(__name__)
tracer = configure_tracing(__file__)


# Wrapper class for LLM Open AI model
class AIModel:
    def __init__(self, azure_deployment, openai_api_version, azure_endpoint, api_key, model_parameters: dict) -> None:
        logger.info("AIModel.Initializing AIModel")
        logger.info(f"AIModel.Initializing:model_parameters = {model_parameters}")

        # TODO add try catch block to catch the exception

        with tracer.start_as_current_span("AIModel.Initializing AIModel") as span:
            self._llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                openai_api_version=openai_api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                temperature=model_parameters.get("temperature", 0),
            )

    def llm(self) -> AzureChatOpenAI:
        return self._llm

if __name__ == "__main__":
    
    # Initialize the AI model
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")

    llm = AIModel(azure_deployment, openai_api_version,
                  azure_endpoint, api_key, {"temperature": 0.5})
