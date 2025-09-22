# rag_core/llm.py
from langchain.chat_models import init_chat_model
from rag_demo.rag_core.config import SETTINGS

def get_llm(*, temperature: float = 0.2, max_tokens: int | None = 512):
    """
    Returns an Azure OpenAI ChatModel ready for LCEL.
    - Low temperature => more stable responses (great for RAG)
    - max_tokens limits output length (None = use deployment default)
    Requires in .env:
      AZURE_OPENAI_ENDPOINT_LLM
      AZURE_OPENAI_KEY
      AZURE_API_VERSION_LLM
      AZURE_OPENAI_DEPLOYMENT_LLM
    """
    # Quick environment validation (clear errors)
    if not SETTINGS.az_llm_endpoint or not SETTINGS.az_key or not SETTINGS.az_api_version_llm or not SETTINGS.az_llm_deployment:
        raise RuntimeError(
            "Missing Azure LLM config: check AZURE_OPENAI_ENDPOINT_LLM, "
            "AZURE_OPENAI_KEY, AZURE_API_VERSION_LLM, AZURE_OPENAI_DEPLOYMENT_LLM."
        )

    # init_chat_model builds the LangChain wrapper for Azure OpenAI provider
    llm = init_chat_model(
        SETTINGS.az_llm_deployment,
        model_provider="azure_openai",
        api_version=SETTINGS.az_api_version_llm,
        api_key=SETTINGS.az_key,
        azure_endpoint=SETTINGS.az_llm_endpoint,
        # Most common generation parameters:
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm
