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
    print("\n[Context selected by retriever]")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", f"doc{i}")
        snippet = d.page_content.replace("\n", " ")[:max_chars]
        print(f"  {i}. [{src}] {snippet}{'...' if len(d.page_content) > max_chars else ''}")
    print()

def build_pipeline(args):
    # Rebuild index if requested
    if args.rebuild:
        p = Path(SETTINGS.persist_dir)
        if p.exists():
            shutil.rmtree(p)
            print(f"[INFO] Removed index folder: {p}")

    # Load components
    embeddings = get_embeddings()
    llm = get_llm(temperature=args.temp, max_tokens=args.max_tokens)

    # Documents + index
    docs = load_real_documents_from_folder(args.data_dir)
    vs = load_or_build_vectorstore(embeddings, docs)

    # Runtime override of retriever parameters
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

    # Complete chain
    chain = build_rag_chain(llm, retriever, prompt)
    return retriever, chain

def main():
    parser = argparse.ArgumentParser(
        description="RAG demo (CLI): ask questions from terminal, show chunks and respond with citations."
    )
    parser.add_argument("--data-dir", default="data", help="Folder with .md/.txt files (default: data)")
    parser.add_argument("--rebuild", action="store_true", help="Delete index and rebuild from scratch")
    parser.add_argument("--k", type=int, default=None, help="Override retriever K (default: from SETTINGS)")
    parser.add_argument("--mmr", action="store_true", help="Use MMR instead of similarity")
    parser.add_argument("--lambda", dest="mmr_lambda", type=float, default=None, help="MMR lambda (0=diversity, 1=relevance)")
    parser.add_argument("--fetch-k", type=int, default=None, help="MMR fetch_k (initial candidates)")
    parser.add_argument("--policy", choices=["strict", "helpful"], default="strict", help="System prompt policy")
    parser.add_argument("--q", "--question", dest="question", help="Single question to ask (if absent enters interactive mode)")
    parser.add_argument("--show-context", action="store_true", help="Print selected chunks before answer")
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

    # Mode 1: question passed as flag
    if args.question:
        ask_once(args.question)
        return

    # Mode 2: interactive from terminal
    print("Interactive mode. Write your question and press Enter.")
    print("Commands: 'exit' to quit, empty input to quit.\n")
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q or q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        ask_once(q)

if __name__ == "__main__":
    main()
