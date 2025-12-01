from smolagents import CodeAgent, OpenAIModel
from agent.tools.smol_rag_tool import RAGQueryTool
from agent.tools.smol_eda_tool import EDASummaryTool
from dotenv import load_dotenv
import os

load_dotenv()

def build_agent():
    # 1. Tools Íinstanziieren
    # Hinweis: Pfade müssen von dort stimmen, wo du main.py ausführst
    rag_tool = RAGQueryTool("./data") 
    eda_tool = EDASummaryTool("./data/nasdaq_100_final_for_RAG.csv")

    # 2. Model definieren (Engine)
    model = OpenAIModel(
        model_id="gpt-4o", # oder gpt-3.5-turbo
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 3. Agent bauen (Driver)
    # CodeAgent ist mächtiger für DS als ToolCallingAgent
    agent = CodeAgent(
        tools=[rag_tool, eda_tool], 
        model=model,
        add_base_tools=True, # Erlaubt dem Agenten, print() und Python-Logik zu nutzen
        additional_authorized_imports=["pandas", "matplotlib", "seaborn"]
    )

    return agent