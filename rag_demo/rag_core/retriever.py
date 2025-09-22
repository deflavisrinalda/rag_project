# rag_core/retriever.py
from .config import SETTINGS

def make_retriever(vector_store):
    """
    Returns a configured retriever:
      - similarity: top-k by similarity
      - mmr: balances relevance and diversity
    """
    if SETTINGS.search_type.lower() == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": SETTINGS.k,               # how many final chunks
                "fetch_k": SETTINGS.fetch_k,   # how many initial candidates
                "lambda_mult": SETTINGS.mmr_lambda,  # 0=diversity, 1=relevance
            },
        )
    # default: similarity
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SETTINGS.k},
    )

