# embeddings.py
from langchain_openai import AzureOpenAIEmbeddings
from rag_demo.rag_core.config import SETTINGS  # environment variables loaded in dataclass

def get_embeddings():
    """
    Returns an AzureOpenAIEmbeddings instance configured from environment variables.
    Used for both indexing (embed_documents) and query (embed_query).
    """
    return AzureOpenAIEmbeddings(
        model=SETTINGS.az_emb_deployment,
        api_key=SETTINGS.az_key,
        azure_endpoint=SETTINGS.az_emb_endpoint,
        api_version=SETTINGS.az_api_version_emb,
    )

