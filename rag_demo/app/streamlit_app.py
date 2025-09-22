# app/streamlit_app.py
import streamlit as st
from pathlib import Path

from rag_demo.rag_core.config import SETTINGS
from rag_demo.rag_core.embeddings import get_embeddings
from rag_demo.rag_core.llm import get_llm
from rag_demo.rag_core.loaders import load_real_documents_from_folder
from rag_demo.rag_core.vectorstore import load_or_build_vectorstore
from rag_demo.rag_core.retriever import make_retriever
from rag_demo.rag_core.prompts import make_prompt, STRICT_CONTEXT_SYSTEM
from rag_demo.rag_core.chain import build_rag_chain, rag_answer

# ---------- Cache di componenti lenti ----------
@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return get_embeddings()

@st.cache_resource(show_spinner=False)
def _get_llm(_temperature: float = 0.2, _max_tokens: int = 512):
    return get_llm(temperature=_temperature, max_tokens=_max_tokens)

@st.cache_data(show_spinner=False)
def _load_docs(_data_dir: str = "data"):
    return load_real_documents_from_folder(_data_dir)

@st.cache_resource(show_spinner=True)
def _build_pipeline(_data_dir: str = "data"):
    """Costruisce l'intera pipeline RAG e la cachea"""
    emb = get_embeddings()
    llm = get_llm(temperature=0.2, max_tokens=512)
    docs = load_real_documents_from_folder(_data_dir)
    vs = load_or_build_vectorstore(emb, docs)
    retriever = make_retriever(vs)
    prompt = make_prompt(STRICT_CONTEXT_SYSTEM)
    chain = build_rag_chain(llm, retriever, prompt)
    return retriever, chain

# ---------- UI ----------
st.set_page_config(page_title="RAG Demo", page_icon="🔎", layout="centered")

st.title("🔎 RAG Demo — domanda → risposta con citazioni")

with st.spinner("Inizializzo il motore..."):
    retriever, chain = _build_pipeline("data")

question = st.text_input("Scrivi la tua domanda", placeholder="Es. Come contattare le RSU in azienda?")

if st.button("Chiedi", type="primary", disabled=not bool(question.strip())):
    with st.spinner("Cerco nei documenti e genero la risposta..."):
        answer = rag_answer(question.strip(), chain)
    st.subheader("Risposta")
    st.write(answer)

    # (facoltativo) mostra i chunk selezionati — utile per capire il grounding
    with st.expander("Vedi contesto selezionato (chunk)"):
        selected_docs = retriever.invoke(question.strip())
        for i, d in enumerate(selected_docs, 1):
            src = d.metadata.get("source", f"doc{i}")
            st.markdown(f"**{i}. [source:{src}]**")
            st.write(d.page_content)
            st.divider()
