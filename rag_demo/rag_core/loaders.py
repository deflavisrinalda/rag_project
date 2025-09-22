# rag_core/loaders.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

SUPPORTED_EXTS = {".txt", ".md"}  # extensible

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Recursively reads .txt/.md files from folder_path.
    For each file:
      - opens with UTF-8 encoding
      - creates Document with page_content and metadata['source'] = filename
    Returns: list of Documents.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Non-existent folder or not a directory: {folder_path}")

    documents: List[Document] = []

    # glob("**/*") = recursive; sort for reproducibility
    for fp in sorted(folder.glob("**/*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue

        # TextLoader handles reading and creates Document
        loader = TextLoader(str(fp), encoding="utf-8")
        docs = loader.load()  # typically 1 document per file

        # add 'source' metadata for citations and traceability
        for d in docs:
            d.metadata["source"] = fp.name  # or str(fp.relative_to(folder))
        documents.extend(docs)

    if not documents:
        raise ValueError(
            f"No documents loaded from '{folder_path}'. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )
    return documents