"""
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.utils import (configure_aoai_env,
                         configure_logging,
                         configure_embedding_env,
                         configure_docintell_env,
                         get_credential)
from io import BytesIO
from aisearch.ai_search import AISearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv
_ = load_dotenv()

# from langchain_community.vectorstores.azuresearch import AzureSearch


embedding_env = configure_embedding_env()
aoai_env = configure_aoai_env()
docintel_env = configure_docintell_env()


def process_document(doc_path, mode="markdown"):
    """
    Process a document using Azure AI Document Intelligence and split it into chunks based on markdown headers.
    """
    # Load the document using Azure AI Document Intelligence.
    loader = AzureAIDocumentIntelligenceLoader(file_path=doc_path,
                                               api_key=docintel_env["doc_intelligence_key"],
                                               api_endpoint=docintel_env["doc_intelligence_endpoint"],
                                               api_model="prebuilt-layout",
                                               api_version=docintel_env["doc_intelligence_api_version"],
                                               mode=mode)
    docs = loader.load()

    # Split the document into chunks base on markdown headers.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)

    doc_content = docs[0].page_content
    chunks = text_splitter.split_text(doc_content)
    print("Length of splits: " + str(len(chunks)))
    return chunks


def get_files_from_blob_storage(storage_account_name: str, container_name: str, folder_name: str) -> List[str]:
    # Create a blob service client
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=DefaultAzureCredential())
    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    # List the blobs in the container
    blobs = container_client.list_blobs(name_starts_with=folder_name)
    # Get the blob names
    files = [blob.name for blob in blobs]
    return files


def download_blob(storage_account_name: str, container_name: str, blob_name: str, download_path: str):
    """
    Download a blob from Azure Blob Storage to a local file.
    """
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=credential
    )
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)

    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"Downloaded {blob_name} to {download_path}")


def download_blob_to_memory(storage_account_name: str, container_name: str, blob_name: str):
    """
    Download a blob from Azure Blob Storage into memory.
    """
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=credential
    )
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)

    blob_data = blob_client.download_blob().readall()
    print(f"Downloaded {blob_name} into memory")
    return BytesIO(blob_data)


# create a function getting a list of files from the blob storage and then it calculates path to the file and
# calls process_document function to process the document
def ingest_files_from_blob(aisearch: AISearch, storage_account_name: str, container_name: str, folder_name: str = None, mode: str = "markdown"):
    """
    Download and process files from Azure Blob Storage.
    """
    files = get_files_from_blob_storage(
        storage_account_name, container_name, folder_name)
    for file in files:
        local_path = os.path.join("/tmp", os.path.basename(file))
        download_blob(storage_account_name, container_name, file, local_path)
        print(f"Processing file: {local_path}")
        chunks = process_document(local_path, mode)
        print(f"Number of chunks: {len(chunks)}")

        aisearch.ingest(chunks)
        os.remove(local_path)  # Clean up the downloaded file
    return chunks

# write a function which loads files from local folder and ingests them to the search index


def ingest_files_from_local_folder(aisearch: AISearch, folder_path: str, mode: str = "markdown"):
    """
    Load and process files from a local folder.
    """
    files = os.listdir(folder_path)
    if len(files) == 0:
        print("No files found in folder")
        return

    print(f"Files in folder>>>: {files}")
    for file in files:
        local_path = os.path.join(folder_path, file)
        print(f"Processing file: {local_path}")
        chunks = process_document(local_path, mode)
        print(f"Number of chunks: {len(chunks)}")

        aisearch.ingest(chunks)
        # os.remove(local_path)  # Clean up the downloaded file
    return chunks


if __name__ == "__main__":

    # Run this script to ingest files from blob storage or from local disk to the search index.
    # AI Search index will be created automatically if it does not exist.

    # If you need a new index to be created, update your parameters accordingly.
    #
    # Example usage:
    # python semantic_chunking_di.py \
    #   --index_name "myIndexName" \
    #   --index_semantic_configuration_name "mySemanticConfigName"

    # example usage:
    # python semantic_chunking_di.py --index_name xx-index1 --index_semantic_configuration_name vector-llmops-workshop-index-semantic-configuration --input_folder /Users/vladfeigin/myprojects/dai-demos/aidemos/llmops/data_preparation/data

    try:

        embedding_env = configure_embedding_env()
        aoai_env = configure_aoai_env()
        docintel_env = configure_docintell_env()

        parser = argparse.ArgumentParser(
            description="Run the script with specified parameters for embedding and indexing.",
            add_help=True
        )

        parser.add_argument("--index_name", required=True,
                            help="Name of the index to be used")
        parser.add_argument("--index_semantic_configuration_name",
                            required=False, help="Semantic configuration for the index")
        parser.add_argument("--input_folder", required=True,
                            help="Input folder path")
        args = parser.parse_args()

        aoai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_env["embedding_model_name"],
            azure_endpoint=embedding_env["embedding_model_endpoint"],
            api_key=aoai_env["api_key"],
            openai_api_version=aoai_env["api_version"],
        )
        vector = aoai_embeddings.embed_query("Hello world")
        print(f" first 10 dimensions : {vector[:10]}")

        # Initialize AISearch instance
        aisearch = AISearch(
            embedding_deployment=embedding_env["embedding_model_name"],
            embedding_endpoint=embedding_env["embedding_model_endpoint"],
            index_name=args.index_name,
            index_semantic_configuration_name=args.index_semantic_configuration_name if args.index_semantic_configuration_name else None
        )
    except Exception as ex:
        print(f"Error initializing AISearch instan: {ex}")

    # Ingest files from local folder (adjust path as needed)
    # chunks = ingest_files_from_local_folder(aisearch,  "../../../data")
    chunks = ingest_files_from_local_folder(aisearch,  args.input_folder)
