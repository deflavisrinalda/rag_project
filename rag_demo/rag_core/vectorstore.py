# rag_core/vectorstore.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag_demo.rag_core.config import SETTINGS
from rag_demo.rag_core.splitter import split_documents

def build_faiss_vectorstore(chunks, embeddings):
    """
    Creates a FAISS vector store from chunks and saves it to disk.
    Returns the FAISS instance ready for .as_retriever().
    """
    vs = FAISS.from_documents(chunks, embedding=embeddings)  # compute & index vectors
    out_dir = Path(SETTINGS.persist_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))  # writes index.faiss + index.pkl
    return vs

def load_or_build_vectorstore(embeddings, docs):
    """
    If a saved index exists, loads it; otherwise does split+build and saves.
    """
    p = Path(SETTINGS.persist_dir)
    index_file = p / "index.faiss"
    meta_file  = p / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # WARNING: loads a pickle -> ok only in trusted environments!
        return FAISS.load_local(
            str(p),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Doesn't exist: create chunks and build index
    chunks = split_documents(docs)
    return build_faiss_vectorstore(chunks, embeddings)
