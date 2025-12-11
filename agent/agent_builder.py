import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

# Import your custom tools
# Ensure these files are in your Python path
from agent.tools.smol_rag_tool_niklas import RAGGraphTool
from agent.tools.smol_eda_tool_max import EDASummaryTool
from agent.tools.smol_image_tool_lasse import ImageGenerationTool

load_dotenv()

# --- THE PERSONA (SYSTEM PROMPT) ---
# This defines the "brain" and personality of your agent.
SMOL_QUANT_PROMPT = """
You are 'Smol-Quant', an elite, autonomous financial analyst agent. 
Your goal is to provide deep, data-driven market insights using a ReAct (Reasoning + Acting) approach.

**YOUR TOOLS:**
1. `financial_analyst` (RAG): Use this for textual analysis, news, and specific company metrics (PE Ratio, Sentiment).
2. `eda_summary` (Data Analysis): Use this to understand the structure of the CSV data available to you.
3. `image_generation_tool`: Use this ONLY to visualize "Market Sentiment" or "Psychology" artistically.
4. **Python Code (Native)**: You can write and execute Python code (pandas, matplotlib) to calculate correlations, volatility, or plot charts from the CSV data.

**YOUR GUIDELINES:**
- **Be Professional:** Adopt a sober, Wall-Street tone. Avoid slang.
- **Data First:** Never guess. If you need numbers, look them up or calculate them using Python.
- **Visuals:** When answering questions about trends or comparisons, ALWAYS try to generate a Python plot (matplotlib).
- **Citations:** When using the RAG tool, mention the source of the news if available.
- **Safety:** DO NOT provide financial advice (e.g., "Buy this stock"). focus on *analysis* and *risk assessment*.

**PLANNING:**
Before answering, think step-by-step:
1. What data do I need? (News vs. Hard Numbers)
2. Do I need to write code to visualize this?
3. Execute the tools.
4. Synthesize the answer.
"""

def build_agent():
    # 1. Model
    model = OpenAIModel(
        model_id="gpt-4o-mini", 
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 2. Tools
    rag_tool = RAGGraphTool(chroma_path="./data/chroma_db")
    eda_tool = EDASummaryTool(csv_path="./data/nasdaq_100_final_for_RAG.csv")
    image_tool = ImageGenerationTool()

    # 3. Agent
    # FIX: 'system_prompt' aus dem Konstruktor entfernt, da es den Fehler verursachte
    agent = CodeAgent(
        tools=[rag_tool, eda_tool, image_tool], 
        model=model,
        add_base_tools=False, 
        additional_authorized_imports=["pandas", "matplotlib", "seaborn", "numpy", "io", "base64", "plotly"],
        max_steps=12,
        verbosity_level=1
    )
    
   # --- FIX FÜR DEN SYSTEM PROMPT ---
    # Wir greifen direkt auf das Template-Dictionary zu, wie die Fehlermeldung es verlangt hat.
    # WICHTIG: Wir holen den originalen Prompt und kleben unsere Persona DAVOR.
    # Würden wir ihn ersetzen, wüsste der Agent nicht mehr, wie er Code schreiben soll.
    
    original_prompt = agent.prompt_templates.get("system_prompt", "")
    agent.prompt_templates["system_prompt"] = SMOL_QUANT_PROMPT + "\n\n" + original_prompt

    return agent