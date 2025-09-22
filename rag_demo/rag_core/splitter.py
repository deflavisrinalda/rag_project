# rag_core/splitter.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag_demo.rag_core.config import SETTINGS

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split ricorsivo per caratteri con priorità di separatori.
    Usa chunk_size e chunk_overlap da SETTINGS.
    Ritorna una nuova lista di Document (uno per chunk) mantenendo i metadata originali.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=[
            "\n\n",  # paragrafi
            "\n",    # righe
            ". ", "? ", "! ",  # frasi
            "; ", ": ", ", ",  # frasi lunghe con punteggiatura
            " ",               # parole
            ""                 # fallback: taglio duro
        ],
    )
    return splitter.split_documents(docs)


