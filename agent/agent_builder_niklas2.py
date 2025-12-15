"""
Agent Builder Module.

Constructs and configures the autonomous financial agent ('Smol-Quant').
Integrates RAG, EDA, and Image Generation tools into a cohesive CodeAgent.
"""
import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

# Import custom tools
from agent.tools.smol_rag_tool import RAGGraphTool
from agent.tools.smol_eda_tool import EDASummaryTool
from agent.tools.smol_image_tool import ImageGenerationTool

# Load environment variables (API keys)
load_dotenv()

# --- SYSTEM PROMPT ---
SMOL_QUANT_PROMPT = """
You are 'Smol-Quant', an autonomous financial analyst agent.
Your goal: Provide deep, data-driven market insights using ReAct (Reasoning + Acting).

**TOOLS & PROTOCOLS:**
1. DATA ACCESS:
   - Use `eda_summary` to inspect CSV columns and find data.
   - Always load CSV via `pd.read_csv('./data/nasdaq_100_final_for_RAG.csv')`.
   - NEVER invent data; if missing, state clearly: "Data not available in source CSV."
2. TEXTUAL ANALYSIS (RAG):
   - Use `financial_analyst` for news summaries and sentiment context.
3. DATA VISUALIZATION:
   - Use Python (`matplotlib`, `seaborn`) for charts, save via `plt.savefig`.
4. IMAGE GENERATION:
   - Use `image_generation_tool` for metaphorical/emotional visualizations.
   - NEVER use it for charts; only for illustrative sentiment images.
5. COMPLIANCE:
   - Analyst role, no financial advice.
   - Forbidden phrases: "Investors should...", "Buy/sell now".
   - Allowed phrasing: "Data indicates...", "Current sentiment is..."
6. HYBRID RESPONSE:
   - Combine tools when answering: RAG + Pandas + Visualization + Image (if appropriate).

**EXECUTION PLAN:**
1. Explore: `eda_summary`
2. Gather: `financial_analyst` + CSV
3. Synthesize: Combine info
4. Visualize: Chart or image
5. Review: Compliance check
"""

def build_agent():
    """
    Builds the Smol-Quant CodeAgent with RAG, EDA, and ImageGenerationTool.
    Returns:
        CodeAgent: initialized agent ready to run.
    """
    # 1. Initialize model
    model = OpenAIModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 2. Instantiate tools
    rag_tool = RAGGraphTool(chroma_path="./data/chroma_db")
    eda_tool = EDASummaryTool(csv_path="./data/nasdaq_100_final_for_RAG.csv")
    image_tool = ImageGenerationTool()  # Image tool added

    # 3. Build agent
    agent = CodeAgent(
        tools=[rag_tool, eda_tool, image_tool],
        model=model,
        add_base_tools=False,  # Only use project tools
        additional_authorized_imports=[
            "pandas", "numpy", "matplotlib", "seaborn",
            "io", "base64", "plotly", "matplotlib.pyplot"
        ],
        max_steps=5,
        verbosity_level=1
    )

    # 4. Inject system prompt
    original_prompt = agent.prompt_templates.get("system_prompt", "")
    agent.prompt_templates["system_prompt"] = SMOL_QUANT_PROMPT + "\n\n" + original_prompt

    return agent
