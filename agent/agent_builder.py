"""
Agent Builder Module.

This module is responsible for constructing and configuring the autonomous
financial agent ('Smol-Quant'). It integrates various tools (RAG, EDA, Image Generation)
into a cohesive CodeAgent architecture. The builder applies a specific persona
and set of operational protocols via prompt injection to ensure the agent adheres
to financial analysis standards and safety guidelines.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

# Import custom tool implementations.
# These tools provide specific capabilities: RAG for text, EDA for structured data,
# and Image Generation for visual sentiment analysis.
from agent.tools.smol_rag_tool_niklas import RAGGraphTool
from agent.tools.smol_eda_tool_max import EDASummaryTool
from agent.tools.smol_image_tool_lasse import ImageGenerationTool

# Load environment variables (e.g., API keys)
load_dotenv()

# --- SYSTEM PROMPT DEFINITION ---
# This constant defines the cognitive framework for the agent. It effectively
# programs the 'Persona' and the 'Operating Procedures' of the LLM.
# It uses a structured approach (Role, Tools, Protocols, Execution Plan) to
# minimize hallucinations and ensure compliance.
SMOL_QUANT_PROMPT = """
You are 'Smol-Quant', an elite, autonomous financial analyst agent.
Your goal is to provide deep, data-driven market insights using a ReAct (Reasoning + Acting) approach.

**YOUR TOOLBOX & PROTOCOLS:**

1.  **DATA ACCESS (CRITICAL - ANTI-HALLUCINATION):**
    * **Step 1 (Metadata):** Use `eda_summary` ONLY to inspect column names and identify the file path.
    * **Step 2 (Loading):** To answer ANY numerical question, YOU MUST WRITE CODE to load the data yourself:
        `df = pd.read_csv('./data/nasdaq_100_final_for_RAG.csv')`
    * **Rule:** NEVER invent, mock, or dummy data. If a company (like Tesla) is not in the CSV after loading it, state clearly: "Data not available in source CSV."

2.  **TEXTUAL ANALYSIS (RAG):**
    * Use the tool `financial_analyst` to find qualitative information: news summaries, sentiment analysis, and context behind price moves.

3.  **DATA VISUALIZATION (CHARTS & GRAPHS):**
    * **When to use:** For accurate comparisons of metrics (e.g., "Volatility of A vs B", "PE Ratio ranking").
    * **Tool:** Use native Python code (`matplotlib.pyplot`, `seaborn`).
    * **CRASH PREVENTION:** NEVER use `plt.show()`. It will crash the runtime.
    * **MANDATORY SAVING:** You must save the plot to a file:
        `plt.savefig('final_plot.png')`
        `plt.close()`
    * Inform the user: "I have generated the data chart."

4.  **ARTISTIC VISUALIZATION (IMAGE GEN):**
    * **When to use:** ONLY for metaphorical, emotional, or illustrative requests (e.g., "Visualize the fear in the market", "Draw a bull run for Nvidia", "Show the sentiment as an image").
    * **Tool:** Use the `image_generation_tool`.
    * **Rule:** NEVER use this tool for creating data charts (like bar graphs). Use Python for that.

5.  **COMPLIANCE & TONE (SAFETY):**
    * **Role:** You are an analyst, NOT a financial advisor.
    * **Forbidden Phrases:** "Investors should...", "Good time to buy/sell", "Strong potential for growth".
    * **Allowed Phrasing:** "The data indicates...", "Historical volatility is...", "Current sentiment is classified as..."
    * **Tone:** Professional, concise, data-focused.

6.  **HYBRID RESPONSE STRATEGY (THE "PRO" MOVE):**
    * If a user asks a general question (e.g., "What do you know about Tesla?", "Analyze Nvidia"), NEVER provide text only.
    * **ALWAYS** combine tools:
        1.  **RAG:** Get the business summary and news.
        2.  **Pandas:** Load the CSV and extract Current Price, Market Cap, and PE Ratio.
        3.  **Visualization:** Plot the key metrics or generate a sentiment image if appropriate.
    * A complete answer MUST have: Text Context + Hard Numbers + A Visual.

**EXECUTION PLAN:**
1.  **Exploration:** Use `eda_summary` to locate data.
2.  **Gathering:** Run `financial_analyst` (Text) AND `pd.read_csv` (Numbers).
3.  **Synthesizing:** Combine both into a comprehensive report.
4.  **Visualizing:** Create a plot (`plt.savefig`) to support your data.
5.  **Review:** Check compliance (No advice!) before final answer.
"""

def build_agent():
    """
    Constructs and configures the Smol-Quant CodeAgent.

    This function initializes the LLM engine, instantiates the necessary tools
    with their correct file paths, and configures the agent's authorized imports
    and system prompts.

    Returns:
        CodeAgent: An initialized agent instance ready for execution.
    """
    
    # 1. Initialize the Model (The Reasoning Engine)
    # GPT-4o-mini is chosen for a balance of speed and code-generation capability.
    model = OpenAIModel(
        model_id="gpt-4o-mini", 
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 2. Instantiate Tools
    # Note: Paths must be relative to the project root where execution occurs.
    rag_tool = RAGGraphTool(chroma_path="./data/chroma_db")
    eda_tool = EDASummaryTool(csv_path="./data/nasdaq_100_final_for_RAG.csv")
    image_tool = ImageGenerationTool()

    # 3. Build the Agent
    # We use CodeAgent to allow the LLM to write and execute Python code directly.
    agent = CodeAgent(
        tools=[rag_tool, eda_tool, image_tool], 
        model=model,
        add_base_tools=False, # Disable default tools (e.g., web search) to enforce strict tool usage.
        # Whitelist libraries to allow data manipulation and plotting within the sandbox.
        additional_authorized_imports=[
            "pandas", 
            "matplotlib", 
            "seaborn", 
            "numpy", 
            "io", 
            "base64", 
            "plotly", 
            "matplotlib.pyplot"
        ],
        max_steps=5, # Limit execution steps to prevent infinite loops.
        verbosity_level=1
    )
    
    # 4. Inject System Prompt (Prompt Engineering)
    # We access the internal prompt template to prepend our custom persona
    # while preserving the agent's base instructions for code generation.
    original_prompt = agent.prompt_templates.get("system_prompt", "")
    agent.prompt_templates["system_prompt"] = SMOL_QUANT_PROMPT + "\n\n" + original_prompt

    return agent