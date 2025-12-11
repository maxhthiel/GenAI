# 🤖 Smol-Quant: Autonomous Financial Analyst Agent

## 📖 Project Overview

**Smol-Quant** is an autonomous agentic system designed to simulate the workflow of a junior financial analyst. Developed using the Hugging Face `smolagents` framework, it moves beyond standard text-prediction chatbots by actively executing code and interacting with disparate data sources.

The system bridges the gap between quantitative financial analysis and qualitative market research by integrating three core capabilities:

1.  **Code Execution:** Autonomous generation and execution of Python scripts for statistical analysis and visualization.
2.  **Semantic Search (RAG):** Retrieval of unstructured qualitative data (news, reports) via vector database querying.
3.  **Visual Synthesis:** Generation of visual representations for market sentiment using generative AI.

## 📂 Project Structure

The project follows a modular architecture, separating agent logic, tool definitions, and data storage.

```text
GenAI/
├── agent/
│   ├── agent_builder.py       # Configuration: Assembles the Agent, Persona, and Tools
│   └── tools/                 # Tool Definitions
│       ├── smol_rag_tool.py   # Interface for ChromaDB (News/Text Retrieval)
│       ├── smol_eda_tool.py   # Interface for Pandas DataFrames (Quantitative Analysis)
│       └── smol_image_tool.py # Interface for Generative Image Models (Visuals)
├── data/
│   ├── chroma_db/             # Vector Database (Persisted Embeddings)
│   └── nasdaq_100...csv       # Structured Financial Dataset
├── main.py                    # Entry Point (CLI / Pipeline Execution)
├── app.py                     # Entry Point (Streamlit Web Interface)
└── .env                       # Environment Configuration (API Keys)
```

## 🏗️ Technical Architecture

The system utilizes a **ReAct (Reasoning + Acting)** pattern, orchestrated by a central code-generating LLM.

### 1\. The Core: `CodeAgent`

Acting as the central orchestrator, the `CodeAgent` does not merely output text. It plans a sequence of actions, generates Python code to execute those actions, and interprets the execution results to formulate a final response.

### 2\. The Toolset

The agent is equipped with specialized tools to handle different data modalities:

  * **Quantitative Analysis Tool (`EDASummaryTool`):**

      * **Function:** Provides metadata and access paths for the underlying CSV dataset.
      * **Mechanism:** Enables the agent to write native `pandas` and `matplotlib` code to perform filtering, aggregation, and visualization of financial metrics (e.g., PE Ratio, Volatility).

  * **Qualitative Research Tool (`RAGQueryTool`):**

      * **Function:** Retrieves context-aware text segments from internal documents.
      * **Mechanism:** Utilizes **ChromaDB** for semantic search. User queries are converted into vector embeddings (`text-embedding-3-small`) to retrieve the most relevant news snippets and business summaries.

  * **Visual Synthesis Tool (`ImageGenerationTool`):**

      * **Function:** Visualizes abstract concepts such as market sentiment.
      * **Mechanism:** Uses an LLM to refine prompts based on retrieved news, which are then passed to a diffusion model (DALL-E 3) to generate illustrative imagery.

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.10+
  * OpenAI API Key

### 1\. Clone Repository

```bash
git clone <repo-url>
cd GenAI
```

### 2\. Environment Setup

It is recommended to use a virtual environment.

**macOS / Linux:**

```bash
python3 -m venv genai
source genai/bin/activate
```

**Windows:**

```bash
python -m venv genai
genai\Scripts\activate
```

### 3\. Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4\. Configuration

Create a `.env` file in the root directory:

```ini
OPENAI_API_KEY=sk-proj-xxxxxx...
OPENAI_MODEL=gpt-4o-mini
```

## 🚀 Usage

The system can be run in two modes:

### A. Terminal Interface (CLI)

Best for debugging and viewing the raw "thought process" of the agent.

```bash
python main.py
```

### B. Web Interface (Streamlit)

Provides a user-friendly chat interface with rendered charts and images.

```bash
streamlit run app.py
```

### Example Queries

  * **Quantitative Analysis:**

    > "Compare the volatility of Tesla and Nvidia over the last year."
    > "Plot the PE Ratio distribution of the Tech sector."

  * **Qualitative Research (RAG):**

    > "What are the recent strategic challenges for Apple?"
    > "Summarize the latest news regarding Microsoft's AI investments."

  * **Hybrid Reasoning:**

    > "First, analyze the market cap of Meta. Then, check the latest news to explain recent price movements."

  * **Visual Generation:**

    > "Visualize the current market sentiment for the semiconductor industry."