from smolagents import Tool
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

# LangChain/LangGraph Imports aus dem Notebook (Zellen 9-16)
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Sicherstellen, dass die Umgebungsvariablen geladen sind
load_dotenv()

# --- LangGraph Hilfsfunktionen ---

# Die State-Definition (aus Zelle 12)
class RAGState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str

# Die Knoten des LangGraph (Reformulate, Retrieve, Generate)

def reformulate_query(state):
    # Logik aus Zelle 13
    if not state.get("chat_history"): 
        return state
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # LLM wird hier benötigt

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Formuliere die Frage so um, dass sie ohne Verlauf verständlich ist (Namen statt Pronomen). Nur die Frage."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    new_query = (prompt | llm | StrOutputParser()).invoke({"history": state["chat_history"], "input": state["input"]})
    
    state["input"] = new_query
    return state

def retrieve(state, retriever):
    # Logik aus Zelle 14
    # 'retriever' wird als Argument übergeben, da es zur Initialisierung gehört
    docs = retriever.invoke(state["input"])
    state["context"] = docs
    return state

def generate(state):
    # Logik aus Zelle 15
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # LLM wird hier benötigt

    docs = state["context"]
    history = state.get("chat_history", [])

    prompt_blocks = []

    for doc in docs:
        # Hier die komplette Prompt-Block-Logik aus Zelle 15 (Text + Metadaten)
        block = f"""
        Company: {doc.metadata.get('company', '')} - Ticker: {doc.metadata.get('ticker', '')}

        Business Summary:
        {doc.page_content}

        Finanzkennzahlen & Metadaten:
        Sektor: {doc.metadata.get('sector', '')}, Industrie: {doc.metadata.get('industry', '')}
        Marktkapitalisierung: {doc.metadata.get('market_cap', '')}
        Aktueller Kurs: {doc.metadata.get('current_price', '')}, 52-Wochen-Hoch: {doc.metadata.get('52_week_high', '')}
        KGV (PE Ratio): {doc.metadata.get('pe_ratio', '')}, KGV (Forward PE): {doc.metadata.get('forward_pe', '')}
        Eigenkapitalrendite (ROE): {doc.metadata.get('roe', '')}, 1-Jahres-Rendite: {doc.metadata.get('return_1y', '')}
        Aktuelle News: {doc.metadata.get('latest_news_title', '')} (Sentiment: {doc.metadata.get('sentiment', '')})

        """
        prompt_blocks.append(block)

    context = "\n\n".join(prompt_blocks)

    messages = [{"role": "system","content": (
            "You are a helpful financial assistant who answers questions based on the given context. Analyze all financial data and metadata carefully. Bei den Kennzahlen handelt es sich um EURO"
        )},*history,{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['input']}"}]

    answer = llm.invoke(messages)

    new_history = history + [
        HumanMessage(content=state["input"]),
        AIMessage(content=answer.content)
    ]

    return {
        "answer": answer.content,
        "chat_history": new_history
    }
    
# --- Das Tool, das alles kapselt ---

class RAGGraphTool(Tool):
    name = "financial_analyst"
    description = "PRIMARY source for ALL financial questions. This tool runs a full RAG-Graph to provide comprehensive analysis including sentiment, news, and complex financial metrics from the high-quality database. ALWAYS use this BEFORE any web search for company or stock queries."
    inputs = {
        "question": {
            "type": "string",
            "description": "The financial or company question to analyze."
        }
    }
    output_type = "string"

    def __init__(self, chroma_path: str):
        super().__init__()
        
        # 1. ChromaDB/Retriever Initialisierung
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            collection_name="nasdaq_docs",
            embedding_function=embeddings,
            persist_directory=chroma_path # <-- Dein persist_dir aus dem Notebook
        )
        self.retriever = db.as_retriever(search_kwargs={"k": 5}) # K=5 für Vergleichbarkeit (wie im Notebook)
        
        # 2. Gedächtnis des Tools (Chat History)
        self.global_chat_history = [] 
        
        # 3. LangGraph Kompilierung (aus Zelle 16)
        
        # Wir müssen retrieve und generate in der Kapselung anpassen,
        # da retrieve den 'retriever' und generate die LLM-Aufrufe benötigt.
        # Am einfachsten: Die Knoten als lambdas oder partielle Funktionen definieren.

        def encapsulated_retrieve(state):
             return retrieve(state, self.retriever)

        self.rag_graph = (
            StateGraph(RAGState)
            .add_node("reformulate", reformulate_query)
            .add_node("retrieve", encapsulated_retrieve) # Nutzt das Retriever-Objekt
            .add_node("generate", generate)
            
            .set_entry_point("reformulate")
            .add_edge("reformulate", "retrieve")
            .add_edge("retrieve", "generate")
            .add_edge("generate", END)
            .compile()
        )

    def forward(self, question: str) -> str:
        # Führt den LangGraph aus und aktualisiert die History (Logik aus Zelle 17: ask-Funktion)
        
        result = self.rag_graph.invoke({
            "input": question, 
            "chat_history": self.global_chat_history # Übergibt den aktuellen Verlauf
        })
        
        # Gedächtnis des Tools aktualisieren (nur die letzten 4 Nachrichten)
        self.global_chat_history = result["chat_history"][-4:]
        
        # Gibt die finale Antwort des LangGraph zurück
        return result['answer']