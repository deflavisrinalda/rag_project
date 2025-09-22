# rag_core/llm.py
from langchain.chat_models import init_chat_model
from rag_demo.rag_core.config import SETTINGS

def get_llm(*, temperature: float = 0.2, max_tokens: int | None = 512):
    """
    Ritorna un ChatModel Azure OpenAI pronto per LCEL.
    - temperature bassa => risposte più stabili (ottimo in RAG)
    - max_tokens limita la lunghezza output (None = lascia default del deployment)
    Richiede nel .env:
      AZURE_OPENAI_ENDPOINT_LLM
      AZURE_OPENAI_KEY
      AZURE_API_VERSION_LLM
      AZURE_OPENAI_DEPLOYMENT_LLM
    """
    # Validazione rapida delle env (errori chiari)
    if not SETTINGS.az_llm_endpoint or not SETTINGS.az_key or not SETTINGS.az_api_version_llm or not SETTINGS.az_llm_deployment:
        raise RuntimeError(
            "Config Azure LLM mancante: verifica AZURE_OPENAI_ENDPOINT_LLM, "
            "AZURE_OPENAI_KEY, AZURE_API_VERSION_LLM, AZURE_OPENAI_DEPLOYMENT_LLM."
        )

    # init_chat_model costruisce il wrapper LangChain per il provider Azure OpenAI
    llm = init_chat_model(
        SETTINGS.az_llm_deployment,
        model_provider="azure_openai",
        api_version=SETTINGS.az_api_version_llm,
        api_key=SETTINGS.az_key,
        azure_endpoint=SETTINGS.az_llm_endpoint,
        # Parametri di generazione più comuni:
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm
