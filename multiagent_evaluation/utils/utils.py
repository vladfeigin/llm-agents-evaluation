from opentelemetry.instrumentation.requests import RequestsInstrumentor
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter, AzureMonitorTraceExporter, AzureMonitorMetricExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.ai.inference.tracing import AIInferenceInstrumentor 
from azure.core.settings import settings 
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

import yaml
import logging
import os
from dotenv import load_dotenv
load_dotenv()


# load agent configuration (variant) from the YAML file
def load_agent_configuration(agent_folder: str, agent_config_file: str) -> dict:

    # add check for input arguments
    if not agent_folder or not agent_config_file:
        raise ValueError("Agent folder and agent config file are required.")

    # Get the directory of the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the absolute path to the configuration file
    config_path = os.path.join(project_root, agent_folder, agent_config_file)

    # Load the configuration file
    with open(config_path, 'r') as file:
        try:
            # Parse the YAML content
            config_data = yaml.safe_load(file)
            # Output the resulting dictionary
            # print(config_data)
        except yaml.YAMLError as error:
            print(f"Error parsing agent config YAML file: {error}")
            raise error

    return config_data


def configure_aoai_env():
    # check if all the required environment variables are set than skip the rest of the code:

    if all([#os.environ.get("AZURE_OPENAI_ENDPOINT"),
            #os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            os.environ.get("AZURE_OPENAI_API_VERSION"),
            os.environ.get("AZURE_OPENAI_KEY")]):
        return {
            #"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "api_key": os.environ.get("AZURE_OPENAI_KEY"),
            #"azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
        }

    # Retrieve environment variables with default values or handle missing cases
    #azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    #azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    api_key = os.environ.get("AZURE_OPENAI_KEY")

    if not all([ api_version, api_key]):
        logging.error(
            "One or more Azure OpenAI environment variables are missing.")
        raise Exception("One or more environment variables are missing.")

    model_config = {
        #"azure_endpoint": azure_endpoint,
        "api_key": api_key,
        #"azure_deployment": azure_deployment,
        "api_version": api_version,
    }
    return model_config


# create a function for configuring emnedding model environment it should load env variables from .env file
def configure_embedding_env():
    if all([os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            os.environ.get("AZURE_OPENAI_KEY"),
            os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            ]):
        return {
            "embedding_model_endpoint": os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            "embedding_model_api_key": os.environ.get("AZURE_OPENAI_KEY"),
            "embedding_model_name": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        }

    # Retrieve environment variables with default values or handle missing cases
    embedding_model_endpoint = os.environ.get(
        "AZURE_OPENAI_EMBEDDING_ENDPOINT")
    embedding_model_api_key = os.environ.get("AZURE_OPENAI_KEY")
    embedding_model_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not all([embedding_model_endpoint, embedding_model_api_key, embedding_model_name]):
        logging.error(
            "One or more Embedding Model environment variables are missing.")
        raise Exception("One or more environment variables are missing.")

    embedding_model_config = {
        "embedding_model_endpoint": embedding_model_endpoint,
        "embedding_model_api_key": embedding_model_api_key,
        "embedding_model_name": embedding_model_name,
    }
    return embedding_model_config


# create function for configuring azur ai search environment it should load env variables from .env file
# beore check if all the required environment variables are set than skip the rest of the code:
# if not set the environment variables and return the dictionary with the environment
# variables
def configure_aisearch_env():
    if all([os.environ.get("AZURE_AI_SEARCH_SERVICE_ENDPOINT"),
            os.environ.get("AZURE_SEARCH_KEY"),
            os.environ.get("AZURE_SEARCH_INDEX_NAME"),
            os.environ.get("AZURE_AI_SEARCH_SERVICE_NAME"),
            ]):
        return {
            "azure_search_endpoint": os.environ.get("AZURE_AI_SEARCH_SERVICE_ENDPOINT"),
            "azure_search_api_key": os.environ.get("AZURE_AI_SEARCH_API_KEY"),
            "azure_search_index_name": os.environ.get("AZURE_SEARCH_INDEX_NAME"),
            "azure_search_service_name": os.environ.get("AZURE_AI_SEARCH_SERVICE_NAME"),
        }

    # Retrieve environment variables with default values or handle missing cases
    azure_search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    azure_search_api_key = os.environ.get("AZURE_SEARCH_KEY")
    azure_search_index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")
    azure_search_service_name = os.environ.get("AZURE_SEARCH_SERVICE_NAME")

    if not all([azure_search_endpoint, azure_search_api_key, azure_search_index_name, azure_search_service_name]):
        logging.error(
            "One or more Azure Search environment variables are missing.")
        raise Exception("One or more environment variables are missing.")

    search_config = {
        "azure_search_endpoint": azure_search_endpoint,
        "azure_search_api_key": azure_search_api_key,
        "azure_search_index_name": azure_search_index_name,
        "azure_search_service_name": azure_search_service_name,
    }
    return search_config

# create a function for configuring azure document intelligence environment it should load env variables from .env file


def configure_docintell_env():
    if all([os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            ]):
        return {
            "doc_intelligence_endpoint": os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            "doc_intelligence_key": os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            "doc_intelligence_api_version": os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION"),
        }

    # Retrieve environment variables with default values or handle missing cases
    doc_intelligence_endpoint = os.environ.get(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intelligence_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    doc_intellugence_api_version = os.environ.get(
        "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION")

    if not all([doc_intelligence_endpoint, doc_intelligence_key, doc_intellugence_api_version]):
        logging.error(
            "One or more Document Intelligence environment variables are missing.")
        raise Exception("One or more environment variables are missing.")

    doc_intelligence_config = {
        "doc_intelligence_endpoint": doc_intelligence_endpoint,
        "doc_intelligence_key": doc_intelligence_key,
        "doc_intellugence_api_version": doc_intellugence_api_version,
    }

    return doc_intelligence_config


def get_credential():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential()
    return credential

# for pf tracing see details here: https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/trace-local-sdk?tabs=python
# local prompt flow traces see in: http://127.0.0.1:23337/v1.0/ui/traces/

# def configure_tracing(collection_name: str = "llmops")-> None:
#    os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
#    configure_azure_monitor(collection_name=collection_name)
#    start_trace()

# Create a function for configuring tracing


def configure_tracing(collection_name: str = "llmops-workshop", enable_console_exporter: bool = True):
    # Initialize tracing provider only once

    settings.tracing_implementation = "opentelemetry" 
    
    
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not connection_string:
            raise ValueError(
                "APPLICATIONINSIGHTS_CONNECTION_STRING environment variable is not set.")

        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = connection_string
        os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"

        configure_azure_monitor(collection_name=collection_name)
        AIInferenceInstrumentor().instrument()
        
        # instrument Langchain
        langchain_instrumentor = LangchainInstrumentor()
        if not langchain_instrumentor.is_instrumented_by_opentelemetry:
            langchain_instrumentor.instrument()
        # general tracing configuration
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        traces_exporter = AzureMonitorTraceExporter()
        trace_processor = BatchSpanProcessor(traces_exporter)
        tracer_provider.add_span_processor(trace_processor)

        # Configure console exporter for local debugging
        # if enable_console_exporter:
        #    console_span_processor = BatchSpanProcessor(ConsoleSpanExporter())
        #    tracer_provider.add_span_processor(console_span_processor)
        # external library instrumentation
        RequestsInstrumentor().instrument()

        # LoggingInstrumentor().instrument(set_logging_format=True)
    # return the configured tracer
    return trace.get_tracer(__name__)


def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # to avoid duplicate logging, check the logger has no handlers
    if not logger.handlers:
        logger.info("Configuring logging. Handlres is being added.")
        # Console handler

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # Add handler to the root logger
        logger.addHandler(console_handler)

        # File handler

        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('azure.core').setLevel(logging.WARNING)
    logging.getLogger('azure.core.pipeline').setLevel(logging.WARNING)
    logging.getLogger('azure.core.pipeline.policies').setLevel(logging.WARNING)
    logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(
        logging.WARNING)
    logging.getLogger('opentelemetry.attributes').setLevel(logging.ERROR)
    logging.getLogger(
        'opentelemetry.instrumentation.instrumentor').setLevel(logging.ERROR)
    logging.getLogger('oopentelemetry.trace').setLevel(logging.ERROR)
    logging.getLogger('oopentelemetry.metrics').setLevel(logging.ERROR)

    return logger
