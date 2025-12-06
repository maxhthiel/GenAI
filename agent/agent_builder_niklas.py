# Datei: agent_builder.py

from smolagents import CodeAgent, OpenAIModel
# 1. ÄNDERUNG: Importiere die neue Klasse
from agent.tools.smol_rag_tool import RAGGraphTool 
from agent.tools.smol_eda_tool import EDASummaryTool
from dotenv import load_dotenv
import os

load_dotenv()

def build_agent():
    # 1. Tools instanziieren
    # Hinweis: Pfade müssen von dort stimmen, wo du main.py ausführst
    
    # 2. ÄNDERUNG: Instanziierung der neuen Klasse
    # 3. WICHTIGE ÄNDERUNG: ANPASSEN DES PFADES! 
    # Wenn Ihre DB-Dateien in GenAI/data/chroma_db liegen, muss der Pfad dorthin zeigen.
    rag_tool = RAGGraphTool("./data/chroma_db") 
    
    eda_tool = EDASummaryTool("./data/nasdaq_100_final_for_RAG.csv")

    # 2. Model definieren (Engine)
    model = OpenAIModel(
        model_id="gpt-4o", # oder gpt-3.5-turbo
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3. Agent bauen (Driver)
    agent = CodeAgent(
        tools=[rag_tool, eda_tool], 
        model=model,
        add_base_tools=True, 
        additional_authorized_imports=["pandas", "matplotlib", "seaborn"]
    )

    return agent