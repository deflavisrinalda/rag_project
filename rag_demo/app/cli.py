# app/cli.py
import argparse
import shutil
from pathlib import Path
from typing import List

from rag_demo.rag_core.config import SETTINGS
from rag_demo.rag_core.embeddings import get_embeddings
from rag_demo.rag_core.llm import get_llm
from rag_demo.rag_core.loaders import load_real_documents_from_folder
from rag_demo.rag_core.vectorstore import load_or_build_vectorstore
from rag_demo.rag_core.retriever import make_retriever
from rag_demo.rag_core.prompts import make_prompt, STRICT_CONTEXT_SYSTEM, HELPFUL_BUT_CAUTIOUS_SYSTEM
from rag_demo.rag_core.chain import build_rag_chain, rag_answer

def preview_docs(docs: List, max_chars: int = 220):
    print("\n[Contesto selezionato dal retriever]")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", f"doc{i}")
        snippet = d.page_content.replace("\n", " ")[:max_chars]
        print(f"  {i}. [{src}] {snippet}{'...' if len(d.page_content) > max_chars else ''}")
    print()

def build_pipeline(args):
    # Rebuild indice se richiesto
    if args.rebuild:
        p = Path(SETTINGS.persist_dir)
        if p.exists():
            shutil.rmtree(p)
            print(f"[INFO] Rimossa cartella indice: {p}")

    # Carica componenti
    embeddings = get_embeddings()
    llm = get_llm(temperature=args.temp, max_tokens=args.max_tokens)

    # Documenti + indice
    docs = load_real_documents_from_folder(args.data_dir)
    vs = load_or_build_vectorstore(embeddings, docs)

    # Override runtime dei parametri retriever
    use_mmr = args.mmr or (SETTINGS.search_type.lower() == "mmr")
    if args.k is not None:
        SETTINGS.k = args.k
    if args.fetch_k is not None:
        SETTINGS.fetch_k = args.fetch_k
    if args.mmr_lambda is not None:
        SETTINGS.mmr_lambda = args.mmr_lambda
    SETTINGS.search_type = "mmr" if use_mmr else "similarity"

    retriever = make_retriever(vs)

    # Prompt policy
    system = STRICT_CONTEXT_SYSTEM if args.policy == "strict" else HELPFUL_BUT_CAUTIOUS_SYSTEM
    prompt = make_prompt(system)

    # Chain completa
    chain = build_rag_chain(llm, retriever, prompt)
    return retriever, chain

def main():
    parser = argparse.ArgumentParser(
        description="RAG demo (CLI): fai domande da terminale, mostra i chunk e risponde con citazioni."
    )
    parser.add_argument("--data-dir", default="data", help="Cartella con i .md/.txt (default: data)")
    parser.add_argument("--rebuild", action="store_true", help="Cancella l'indice e ricostruisce da zero")
    parser.add_argument("--k", type=int, default=None, help="Override del K del retriever (default: da SETTINGS)")
    parser.add_argument("--mmr", action="store_true", help="Usa MMR invece di similarity")
    parser.add_argument("--lambda", dest="mmr_lambda", type=float, default=None, help="MMR lambda (0=diversità, 1=pertinenza)")
    parser.add_argument("--fetch-k", type=int, default=None, help="MMR fetch_k (candidati iniziali)")
    parser.add_argument("--policy", choices=["strict", "helpful"], default="strict", help="System prompt policy")
    parser.add_argument("--q", "--question", dest="question", help="Domanda singola da porre (se assente entra in modalità interattiva)")
    parser.add_argument("--show-context", action="store_true", help="Stampa i chunk selezionati prima della risposta")
    parser.add_argument("--temp", type=float, default=0.2, help="LLM temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=512, help="LLM max tokens (default: 512)")
    args = parser.parse_args()

    retriever, chain = build_pipeline(args)

    def ask_once(q: str):
        if args.show_context:
            docs_sel = retriever.invoke(q)
            preview_docs(docs_sel)
        ans = rag_answer(q, chain)
        print("Q:", q)
        print("A:", ans)
        print("-" * 80)

    # Modalità 1: domanda passata come flag
    if args.question:
        ask_once(args.question)
        return

    # Modalità 2: interattiva da terminale
    print("Modalità interattiva. Scrivi la tua domanda e premi Invio.")
    print("Comandi: 'exit' per uscire, invio vuoto per uscire.\n")
    while True:
        try:
            q = input("Domanda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            break
        if not q or q.lower() in {"exit", "quit"}:
            print("Uscita.")
            break
        ask_once(q)

if __name__ == "__main__":
    main()
