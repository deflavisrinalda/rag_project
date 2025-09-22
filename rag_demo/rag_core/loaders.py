# rag_core/loaders.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

SUPPORTED_EXTS = {".txt", ".md"}  # estendibile

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Legge ricorsivamente file .txt/.md da folder_path.
    Per ogni file:
      - apre con encoding UTF-8
      - crea Document con page_content e metadata['source'] = nome file
    Ritorna: lista di Document.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Cartella inesistente o non directory: {folder_path}")

    documents: List[Document] = []

    # glob("**/*") = ricorsivo; ordina per riproducibilità
    for fp in sorted(folder.glob("**/*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue

        # TextLoader gestisce lettura e crea Document
        loader = TextLoader(str(fp), encoding="utf-8")
        docs = loader.load()  # tipicamente 1 document per file

        # aggiungi metadato 'source' per citazioni e tracciabilità
        for d in docs:
            d.metadata["source"] = fp.name  # oppure str(fp.relative_to(folder))
        documents.extend(docs)

    if not documents:
        raise ValueError(
            f"Nessun documento caricato da '{folder_path}'. "
            f"Estensioni supportate: {sorted(SUPPORTED_EXTS)}"
        )
    return documents