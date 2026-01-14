from smolagents import Tool
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

# LangChain/LangGraph imports for orchestration and state management
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables (API Keys) from .env file
load_dotenv()

# LangGraph Helper Functions

# Define the shared state structure for the LangGraph workflow
class RAGState(TypedDict):
    input: str
    context: List[Document]
    answer: str

# Refines the user query to optimize it for vector search
def reformulate_query(state):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Prompt focusing on transforming the input into keyword-rich search terms
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial search queries. Optimise the following user query for a semantic search in a database. Return ONLY the optimised search query."),
        ("human", "{input}")
    ])

    new_query = (prompt | llm | StrOutputParser()).invoke({"input": state["input"]})
    state["input"] = new_query
    return state

# Fetches relevant documents from the ChromaDB vector store
def retrieve(state, retriever):
    docs = retriever.invoke(state["input"])
    state["context"] = docs
    return state

# Synthesizes the final answer using retrieved context and metadata
def generate(state):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 
    docs = state["context"]
    
    prompt_blocks = []

    # Iterate through retrieved docs to build a context block including financial metadata
    for doc in docs:
        block = f"Company: {doc.metadata.get('company', '')} - Ticker: {doc.metadata.get('ticker', '')}\n"
        block += f"Business Summary: {doc.page_content}\n\n"
        
        # Adding all relevant financial metadata as requested
        block += "Finanzkennzahlen & Metadaten\n"
        block += f"Ticker: {doc.metadata.get('ticker', '')}, Unternehmen: {doc.metadata.get('company', '')}\n"
        block += f"Sektor: {doc.metadata.get('sector', '')}, Industrie: {doc.metadata.get('industry', '')}\n"
        block += f"Marktkapitalisierung: {doc.metadata.get('market_cap', '')}\n"
        block += f"Aktueller Kurs: {doc.metadata.get('current_price', '')}, Vortagesschluss: {doc.metadata.get('previous_close', '')}, 52-Wochen-Hoch: {doc.metadata.get('52_week_high', '')}\n"
        block += f"KGV (PE Ratio): {doc.metadata.get('pe_ratio', '')}, KGV (Forward PE): {doc.metadata.get('forward_pe', '')}\n"
        block += f"Dividendenrendite: {doc.metadata.get('dividend_yield', '')}, Kurs-Buchwert-Verhältnis (PB): {doc.metadata.get('price_to_book', '')}\n"
        block += f"Gesamtumsatz: {doc.metadata.get('total_revenue', '')}, Verschuldungsgrad (Debt/Equity): {doc.metadata.get('debt_to_equity', '')}\n"
        block += f"Eigenkapitalrendite (ROE): {doc.metadata.get('roe', '')}, 1-Jahres-Rendite: {doc.metadata.get('return_1y', '')}, durchschnittliche monatl. Rendite der letzten 5 jahre: {doc.metadata.get('avg_monthly_return', '')}\n"
        block += f"Volatilität: {doc.metadata.get('volatility', '')}, Link zur Webseite: {doc.metadata.get('website', '')}\n\n"
        block += f"Nachrichten Titel: {doc.metadata.get('latest_news_title', '')}, Bewertung der News: {doc.metadata.get('sentiment', '')}, Link zur News: {doc.metadata.get('news_link', '')}\n"
        block += f"Inhalt: {doc.metadata.get('news_summary', '')}\n"
        
        prompt_blocks.append(block)

    context = "\n\n".join(prompt_blocks)

    # Final prompt for the Output
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant. Answer based on the context. All metrics are in EURO."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['input']}"}
    ]

    answer = llm.invoke(messages)
    return {"answer": answer.content}
    
# Tool Wrapper Class

class RAGGraphTool(Tool):
    # smolagents tool metadata
    name = "financial_analyst"
    description = "PRIMARY source for qualititive Insights and for all financial questions. This tool runs a full RAG-Graph to provide comprehensive analysis including sentiment, news, and complex financial metrics from the high-quality database. ALWAYS use this BEFORE any web search for company or stock queries."
    inputs = {
        "question": {
            "type": "string",
            "description": "The financial or company question to analyze."
        }
    }
    output_type = "string"

    def __init__(self, chroma_path: str):
        super().__init__()
        
        # Initializing Embeddings and Vector Store connection
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            collection_name="nasdaq_docs",
            embedding_function=embeddings,
            persist_directory=chroma_path 
        )
        # retriever with k=4. After Testing it turns out that k=4 is working well
        self.retriever = db.as_retriever(search_kwargs={"k": 4}) 
        
        # Defines a wrapper for the retrieve node to pass the instance's retriever
        def encapsulated_retrieve(state):
             return retrieve(state, self.retriever)

        # Builds the final Langgraph Workflow
        self.rag_graph = (
            StateGraph(RAGState)
            .add_node("reformulate", reformulate_query)
            .add_node("retrieve", encapsulated_retrieve) 
            .add_node("generate", generate)
            
            # Defines the execution flow
            .set_entry_point("reformulate")
            .add_edge("reformulate", "retrieve")
            .add_edge("retrieve", "generate")
            .add_edge("generate", END)
            .compile()
        )

    def forward(self, question: str) -> str:
        # Executes the graph and returns the result of the "generate" node
        result = self.rag_graph.invoke({"input": question})
        return result["answer"]
