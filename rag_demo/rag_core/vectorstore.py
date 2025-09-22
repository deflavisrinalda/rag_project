# rag_core/vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag_demo.rag_core.config import SETTINGS
from rag_demo.rag_core.splitter import split_documents

def build_faiss_vectorstore(chunks, embeddings):
    """
    Crea un vector store FAISS a partire dai chunk e lo salva su disco.
    Ritorna l'istanza FAISS pronta per .as_retriever().
    """
    vs = FAISS.from_documents(chunks, embedding=embeddings)  # compute & index vectors
    out_dir = Path(SETTINGS.persist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))  # scrive index.faiss + index.pkl
    return vs

def load_or_build_vectorstore(embeddings, docs):
    """
    Se esiste un indice salvato, lo carica; altrimenti fa split+build e salva.
    """
    p = Path(SETTINGS.persist_dir)
    index_file = p / "index.faiss"
    meta_file  = p / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # ATTENZIONE: carica un pickle -> ok solo in ambienti fidati!
        return FAISS.load_local(
            str(p),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Non esiste: crea i chunk e costruisci l'indice
    chunks = split_documents(docs)
    return build_faiss_vectorstore(chunks, embeddings)
