# embeddings.py
from langchain_openai import AzureOpenAIEmbeddings
from rag_demo.rag_core.config import SETTINGS  # dove hai caricato le env nel dataclass

def get_embeddings():
    """
    Restituisce un'istanza di AzureOpenAIEmbeddings configurata dalle env.
    Usata sia per indicizzazione (embed_documents) sia per query (embed_query).
    """
    return AzureOpenAIEmbeddings(
        model=SETTINGS.az_emb_deployment,
        api_key=SETTINGS.az_key,
        azure_endpoint=SETTINGS.az_emb_endpoint,
        api_version=SETTINGS.az_api_version_emb,
    )

