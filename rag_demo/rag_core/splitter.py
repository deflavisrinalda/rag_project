# rag_core/splitter.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag_demo.rag_core.config import SETTINGS

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Recursive character splitting with priority separators.
    Uses chunk_size and chunk_overlap from SETTINGS.
    Returns a new list of Documents (one per chunk) preserving original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=[
            "\n\n",  # paragraphs
            "\n",    # lines
            ". ", "? ", "! ",  # sentences
            "; ", ": ", ", ",  # long sentences with punctuation
            " ",               # words
            ""                 # fallback: hard cut
        ],
    )
    return splitter.split_documents(docs)


