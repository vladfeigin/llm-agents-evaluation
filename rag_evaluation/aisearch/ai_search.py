from logging import INFO, getLogger
from utils.utils import configure_logging
from opentelemetry import trace
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.vectorstores.azuresearch import AzureSearch
import os
import atexit
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Logging calls with this logger will be tracked
logger = configure_logging()
tracer = trace.get_tracer(__name__)

# Azure Search configuration
AZURE_AI_SEARCH_SERVICE_ENDPOINT = os.getenv(
    "AZURE_AI_SEARCH_SERVICE_ENDPOINT")
AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")

# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

_required_env_vars = [
    "AZURE_AI_SEARCH_SERVICE_ENDPOINT", "AZURE_AI_SEARCH_API_KEY",
    "AZURE_OPENAI_KEY", "AZURE_OPENAI_API_VERSION"
]

for var in _required_env_vars:
    if not os.getenv(var):
        logger.error(f"Environment variable {var} is not set.")
        raise EnvironmentError(f"Environment variable {var} is not set.")

# AISearch class to perform search operations
class AISearch:

    # init method to initialize the class
    def __init__(self, embedding_deployment: str,  embedding_endpoint: str, index_name: str, index_semantic_configuration_name: str) -> None:

        logger.info("AISearch.Initializing Azure Search client.")

        logger.info(f"ai search index name : {index_name}")
        self.index_name = index_name
        self.index_semantic_configuration_name = index_semantic_configuration_name

        # Initialize AzureOpenAIEmbeddings
        self._embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            azure_endpoint=embedding_endpoint,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_KEY
        )

        # Define the fields for the index
        self._fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchableField(
                name="chunk",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(
                    self._embeddings.embed_query("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
        ]

        try:

            # Create Langchain AzureSearch object
            self._vector_search = AzureSearch(
                azure_search_endpoint=AZURE_AI_SEARCH_SERVICE_ENDPOINT,
                azure_search_key=AZURE_AI_SEARCH_API_KEY,
                index_name=index_name,
                embedding_function=self._embeddings.embed_query,
                semantic_configuration_name=index_semantic_configuration_name if index_semantic_configuration_name else None,
                additional_search_client_options={
                    "retry_total": 3, "logging_enable": True, "logger": logger},
                fields=self._fields,
            )

            atexit.register(self.__close__)
        except Exception as e:
            logger.error(f"Error during ai search index initialization: {e}")
            raise Exception(
                f"Error during ai search index initialization: {e}")

    def __close__(self) -> None:
        """
        Close the Azure Search client.
        """
        print("Closing Azure Search client.")

    # TODO - add as parameter the search type and the top_k here instead having them as fixed values
    def create_retriever(self, search_type: str, top_k=3) -> AzureAISearchRetriever:
        # Create retriever object
        # supported search types: "semantic_hybrid", "similarity" (default) , "hybryd"
        return self._vector_search.as_retriever(search_type=search_type, k=top_k)

    def ingest(self, documents: list, **kwargs) -> None:
        """
        Ingest documents into Azure Search.

        :param documents: List of document chunks to ingest.
        :param metadata: List of metadata corresponding to each document chunk.
        :raises ValueError: If input is invalid.
        """
        if not isinstance(documents, list) or not documents:
            raise ValueError("Input must be a non-empty list")

        self._vector_search.add_documents(documents)

    # TODO: Add thresholds and output score
    def search(self, query: str, search_type: str = 'hybrid', top_k: int = 5) -> str:
        """
        Search for similar documents in Azure Search.

        :param query: Search query string.
        :param search_type: Type of search to perform.
        :param top_k: Number of top results to return.
        :return: Content of the top search result.
        :raises ValueError: If input is invalid.
        """
        logger.info(
            f"Search: Searching for similar documents using query: {query}")
        with tracer.start_as_current_span("aisearch") as aisearch_span:
            if not isinstance(query, str) or not query:
                raise ValueError("Search query must be a non-empty string")
            aisearch_span.set_attribute("ai_search_query:", query)

            docs = self._vector_search.similarity_search(
                query=query, k=top_k, search_type=search_type)

            # run in loop on the list of documents take the content for each document in page_content and concatenate them. put tab between content of each document
            # return the concatenated content
            # each document in the list is: langchain_core.documents.base.Document
            final_content = ""
            for doc in docs:
                final_content += doc.page_content + "\t"
            return final_content

# Main function to test the AISearch class
if __name__ == "__main__":
    try:
        aisearch = AISearch()
        content = aisearch.search("What Microsoft Fabric",
                                  search_type='hybrid', top_k=3)
        print("Content:>>>> ", content)

        """
        docs_retr = aisearch.retriever().invoke("What is Microsoft Fabric?")
        content = ""
        for doc in docs_retr:
            content += doc.page_content + "###"
            
        print("Content:>>>> ", content)
        """
    except Exception as e:
        logger.error(f"Error during search: {e}")
    finally:
        logger.info("Search completed.")
