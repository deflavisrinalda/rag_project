# rag_core/prompts.py
from langchain_core.prompts import ChatPromptTemplate

# 1) Politiche (system prompts) che puoi alternare in demo
STRICT_CONTEXT_SYSTEM = (
    "Sei un assistente che risponde SOLO con le informazioni presenti nel contesto fornito. "
    "Se l'informazione non è presente, rispondi esattamente: 'Non è presente nel contesto fornito.' "
    "Includi sempre le citazioni nel formato [source:NOME_FILE]. "
    "Rispondi in italiano, in modo conciso e tecnico."
)

HELPFUL_BUT_CAUTIOUS_SYSTEM = (
    "Sei un assistente utile e cauto. Rispondi usando prioritariamente il contesto fornito. "
    "Se una parte della risposta non è nel contesto, dichiaralo esplicitamente e NON inventare. "
    "Includi sempre citazioni per ciò che prendi dal contesto nel formato [source:NOME_FILE]. "
    "Rispondi in italiano, chiaro e sintetico."
)

# 2) Costruttore del prompt
def make_prompt(system: str = STRICT_CONTEXT_SYSTEM) -> ChatPromptTemplate:
    """
    Ritorna un ChatPromptTemplate con:
    - messaggio di sistema (politica)
    - messaggio utente con domanda, contesto e istruzioni esplicite
    """
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni operative:\n"
         "1) Usa SOLO le informazioni nel contesto per rispondere.\n"
         "2) Cita sempre le fonti [source:NOME_FILE] accanto alle affermazioni rilevanti.\n"
         "3) Se l'informazione non è nel contesto, scrivi esattamente: 'Non è presente nel contesto fornito.'")
    ])
