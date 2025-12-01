from smolagents import OpenAIModel
from agent.tools.smol_rag_tool import RAGQueryTool
from agent.tools.smol_eda_tool import EDASummaryTool
from dotenv import load_dotenv
import os

load_dotenv()

def build_agent():
    # Tool Instanz erstellen
    rag_tool = RAGQueryTool("data/nasdaq_chroma_db")
    rag_tool.output_type = str   # <-- setze das nach der Init
    
    eda_tool = EDASummaryTool("data/nasdaq_100_final_for_RAG.csv")
    eda_tool.output_type = str   # <-- setze das nach der Init

    return OpenAIModel(
        model_id=os.getenv("OPENAI_MODEL"),
        tools=[rag_tool, eda_tool],
        max_steps=5,
        verbose=True
    )
