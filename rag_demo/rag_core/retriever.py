# rag_core/retriever.py
from .config import SETTINGS

def make_retriever(vector_store):
    """
    Ritorna un retriever configurato:
      - similarity: top-k per similarità
      - mmr: bilancia pertinenza e diversità
    """
    if SETTINGS.search_type.lower() == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": SETTINGS.k,               # quanti chunk finali
                "fetch_k": SETTINGS.fetch_k,   # quanti candidati iniziali
                "lambda_mult": SETTINGS.mmr_lambda,  # 0=diversità, 1=pertinenza
            },
        )
    # default: similarity
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SETTINGS.k},
    )

