# rag_core/chain.py
from typing import List
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Concatenates chunks with inline citations: [source:<file>].
    Maintains retriever order (most relevant at top).
    """
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", f"doc{i}")
        parts.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(parts)

def build_rag_chain(llm, retriever, prompt):
    """
    LCEL pipeline:
      {question} --> {context=retriever -> format} --> prompt --> llm --> string
    """
    chain = (
        {
            "question": RunnablePassthrough(),         # pass question as-is
            "context": retriever | format_docs_for_prompt,  # retrieve + format
        }
        | prompt  # ChatPromptTemplate
        | llm     # Chat model (Azure)
        | StrOutputParser()  # return clean string
    )
    return chain

def rag_answer(question: str, chain) -> str:
    """
    Convenient wrapper for a single question.
    """
    return chain.invoke(question)

from rag_demo.rag_core.embeddings import get_embeddings
from rag_demo.rag_core.llm import get_llm
from rag_demo.rag_core.loaders import load_real_documents_from_folder
from rag_demo.rag_core.vectorstore import load_or_build_vectorstore
from rag_demo.rag_core.retriever import make_retriever
from rag_demo.rag_core.prompts import make_prompt
from rag_demo.rag_core.chain import build_rag_chain, rag_answer

emb = get_embeddings()
llm = get_llm(temperature=0.0, max_tokens=300)
docs = load_real_documents_from_folder("data")
vs = load_or_build_vectorstore(emb, docs)
retriever = make_retriever(vs)
prompt = make_prompt()  # strict-context by default
chain = build_rag_chain(llm, retriever, prompt)

print(rag_answer("Come contatto le RSU/RSA?", chain))

