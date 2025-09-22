from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    # Persistence
    # Folder where the vector store saves and reads the index
    persist_dir: str = os.getenv("PERSIST_DIR", "index")# Persistence
    # Folder where the vector store saves and reads the indexlasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    # Persistenza
    # Cartella dove il vector store salva e rilegge l’indice
    persist_dir: str = os.getenv("PERSIST_DIR", "index")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))

    # Retrieval
    search_type: str = os.getenv("SEARCH_TYPE", "similarity")  # "similarity" | "mmr"
    k: int = int(os.getenv("RETRIEVER_K", 5))
    fetch_k: int = int(os.getenv("FETCH_K", 20))
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", 0.3))

    # Embeddings/LLM providers
    embedding_provider: str = os.getenv("EMBED_PROVIDER", "azure")  # "azure" | "hf"
    llm_provider: str = os.getenv("LLM_PROVIDER", "azure")          # "azure" | "lmstudio"

    # Credentials
    # Azure embeddings
    az_emb_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "")
    az_emb_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING", "")
    az_api_version_emb: str = os.getenv("AZURE_API_VERSION_EMBEDDING", "")
    az_key: str = os.getenv("AZURE_OPENAI_KEY", "")

    # Azure LLM
    az_llm_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM", "")
    az_llm_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT_LLM", "")
    az_api_version_llm: str = os.getenv("AZURE_API_VERSION_LLM", "")

SETTINGS = Settings()
