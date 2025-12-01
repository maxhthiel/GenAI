# 🤖 Smol-Quant: Autonomous Financial Analyst Agent

## 📂 Project Structure

This is how the project is organized. The **Agent** logic sits in the `agent/` folder, while all **Data** (both raw CSV and the Vector Database) resides in `data/`.

```text
GenAI/
├── agent/
│   ├── agent_builder.py    # The blueprint: Configures the Agent and assigns Tools
│   └── tools/              # The capabilities
│       ├── smol_rag_tool.py   # Tool for searching internal documents
│       └── smol_eda_tool.py   # Tool for analyzing the CSV dataset
├── data/
│   ├── chroma_db/          # The Vector Database (Knowledge Base)
│   └── nasdaq_100...csv    # Raw Financial Data
├── main.py                 # The Entry Point (Run this file)
└── .env                    # Configuration (API Keys)
````

-----

## 📖 Project Overview

**Smol-Quant** is a GenAI-powered agent designed to simulate a junior financial analyst. Unlike standard chatbots that only predict text, Smol-Quant operates as an **agentic system**.

It bridges the gap between quantitative data analysis and qualitative news research by combining:

1.  **Code Execution:** The agent writes and runs real Python code to calculate metrics from data.
2.  **Semantic Search (RAG):** The agent "reads" internal news and reports using a vector database.
3.  **Autonomous Reasoning:** It breaks down complex user questions into logical steps (Plan -\> Execute -\> Answer).

-----

## 🏗️ Technical Architecture (Simplified)

Think of the architecture as a **Brain** equipped with specialized **Tools**.

### 1\. The Brain: `CodeAgent` (Orchestrator)

We use the **Hugging Face `smolagents` framework**.

  * Instead of just answering with text, this agent thinks in **Python**.
  * When you ask a question, the "Brain" writes a Python script to solve it, executes that script securely, and observes the result.

### 2\. The Tools (Capabilities)

The agent has access to two primary sources of truth:

  * **The "Analyst" Tool (`EDASummaryTool`):**

      * **What it does:** Gives the agent direct access to the `pandas` DataFrame containing stock prices, PE ratios, and volatility.
      * **How it works:** The agent writes pandas code (e.g., `df.groupby('Sector').mean()`) to answer quantitative questions accurately.

  * **The "Researcher" Tool (`RAGQueryTool`):**

      * **What it does:** Allows the agent to search through thousands of text snippets (News, Business Summaries).
      * **How it works:** It uses **ChromaDB** (Vector Database). Your question is converted into a mathematical vector (`text-embedding-3-small`), and the database returns the most relevant text segments (Semantic Search).

-----

## ⚙️ Setup & Run

Follow these steps to start the agent.

### 1\. Clone Repository

```bash
git clone <repo-url>
cd GenAI
```

### 2\. Create & Activate Virtual Environment

We use a clean environment named `genai`.

**macOS / Linux:**

```bash
python -m venv genai
source genai/bin/activate
```

**Windows:**

```bash
python -m venv genai
genai\Scripts\activate
```

### 3\. Configure API Keys

Create a file named `.env` in the root folder and add your OpenAI credentials:

```ini
# .env file
OPENAI_API_KEY=sk-proj-xxxxxx...
OPENAI_MODEL=gpt-4o
```

### 4\. Run the Agent

Everything is pre-configured. Start the terminal interface:

```bash
python main.py
```

-----

## 🚀 Usage Guide

Once the terminal interface is running, you can interact naturally. The agent automatically decides whether to calculate numbers or search for text.

**Examples:**

  * **Quantitative Question (Agent writes code):**

    > "Analyze the dataset. Which company has the highest volatility?"
    > "Calculate the average PE Ratio of the Tech sector."

  * **Qualitative/RAG Question (Agent searches DB):**

    > "Nutze die interne Datenbank. Was sind die aktuellen News zu Nvidia?"
    > "Why is the sentiment for Apple negative?"

  * **Combined Reasoning:**

    > "First check the volatility of Tesla, then find reasons for it in the news."


```
```