# 🤖 Smol-Quant: Autonomous Financial Analyst Agent

## 📖 Project Overview

**Smol-Quant** is an autonomous agentic system designed to simulate the workflow of a junior financial analyst. Developed as a capstone project for the "Generative AI" course, this system addresses the fundamental limitations of standard LLMs in financial contexts.

### The Problem

Standard LLMs frequently hallucinate financial data when asked to perform precise calculations or retrieve up-to-date market information. They lack access to verified internal datasets and often fail to distinguish between creative writing and factual reporting.

### The Solution: Grounded Truth

The agent cannot invent numbers. It must retrieve them from two distinct, verified data sources:

1. **Structured Data:** A comprehensive NASDAQ-100 dataset (CSV) for hard metrics like PE Ratio, Volatility, and Market Cap.
2. **Unstructured Data:** A Vector Database (ChromaDB) containing news, business summaries, and context from sources like Wikipedia and Yahoo Finance.

---

## 🏗️ System Architecture

### 1. The Brain: CodeAgent (Orchestrator)

At the core lies the `CodeAgent`. Unlike a simple chatbot, this component acts as a reasoning engine. It writes and executes Python code to solve complex problems.

### 2. The Logic: ReAct Pattern

The agent follows the **ReAct (Reasoning + Acting)** paradigm. For every user query, it autonomously cycles through:

* **Reasoning:** Analyzing the user's intent.
* **Tool Selection:** Deciding which specific tool is required.
* **Observation:** Reading the output of the tool execution to inform the next step.

### 3. The Safety: LLM-as-a-Judge Pipeline

To ensure operational safety, we implemented a **"Compliance Officer"** layer. This secondary model intercepts every draft response before it reaches the user.

* **Compliance Check:** Scans for financial advice violations or hallucinations.
* **Self-Correction Loop:** If the Judge rejects an answer, the feedback is injected back into the agent's memory, forcing it to replan and correct its output automatically.

![Agent Pipeline Architecture](images/Agent.jpeg)

---

## 🛠️ The Toolset

The agent is sandboxed and equipped with three specialized tools to handle different data modalities.

### 1. EDA Tool (`eda_summary`)

* **Function:** Acts as the data scout.
* **Capability:** Provides the agent with metadata, column structures, and statistical summaries of the NASDAQ-100 dataset. This allows the agent to understand the "shape" of the data before performing deep analysis.

### 2. Financial Analyst Tool (`financial_analyst`)

* **Function:** The RAG (Retrieval Augmented Generation) interface.
* **Capability:** Performs semantic searches within the ChromaDB vector store. It retrieves qualitative context—such as recent strategic challenges or leadership changes—to explain the "why" behind the numbers.

![RAG Pipeline Architecture](images/RAG.jpeg)

### 3. Image Generation Tool (`image_generation_tool`)

* **Function:** The visual artist.
* **Capability:** Connects to generative image models (DALL-E 3) to create illustrative visuals for abstract concepts, such as "market sentiment" or "bull runs," adding a multi-modal dimension to the report.

![Image Generation Tool](images/Image.jpeg)
---

## 🔒 Security & Sandbox Environment

To prevent the agent from executing malicious code, the `CodeAgent` operates within a restricted local sandbox. It does not have unrestricted access to the host machine's shell or file system.

### Allowed Libraries
The agent is strictly limited to importing only a specific set of safe libraries required for data analysis and visualization. Any attempt to import unauthorized modules (e.g., `os`, `sys`, `requests`) is blocked by the runtime.

**Authorized Imports:**
* `pandas` (Data Manipulation)
* `numpy` (Numerical Computing)
* `matplotlib.pyplot` & `seaborn` (Data Visualization)
* `io`, `base64`, `json`, `ast` (Data Processing)

---

## 📂 Project Structure

The project follows a modular architecture separating logic, tools, and data.

```text
GenAI/
├── agent/
│   ├── agent_builder.py       # Factory: Assembles the Agent, Persona, and Safety Prompts
│   └── tools/                 # Tool Definitions
│       ├── smol_rag_tool.py   # RAG Interface (Financial Analyst)
│       ├── smol_eda_tool.py   # EDA Interface (Pandas Scout)
│       └── smol_image_tool.py # Visual Interface (Image Gen)
├── data/
│   ├── chroma_db/             # Persistent Vector Store (Embeddings)
│   └── nasdaq_100...csv       # Structured Financial Dataset (Source of Truth)
├── data_ingestions/
│   ├── data_scraping.py       # Scrape and structure data
│   └── embeddings.py          # Create embeddings and RAG pipeline
├── evaluate_agent.py          # Offline Evaluation Pipeline (Scientific Metrics)
├── main.py                    # CLI Entry Point & Compliance Logic
├── app.py                     # Streamlit Web Interface (Production UI)
├── requirements.txt           # Dependency Manifest
└── .env                       # API Key Configuration

```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.10 or higher
* OpenAI API Key (with access to GPT-4o and DALL-E 3)

### 1. Clone Repository

```bash
git clone <repo-url>
cd GenAI

```

### 2. Environment Setup

It is highly recommended to use a virtual environment.

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configuration

Create a `.env` file in the root directory and add your credentials:

```ini
OPENAI_API_KEY=sk-proj-xxxxxx...
# Optional: Model overrides
OPENAI_MODEL=gpt-4o-mini

```

---

## 🚀 Usage

### 1. Web Interface (Streamlit)

The primary way to interact with Smol-Quant is via the dashboard, which supports chart rendering, session memory, and the "Thought Process" visualization.

```bash
streamlit run app.py

```

### 2. Evaluation Pipeline (Scientific Validation)

To run the automated quality audit (LLM-as-a-Judge):

```bash
python evaluate_agent.py

```

*This generates a `evaluation_rich_data.json` with detailed performance metrics.*

---

## 📊 Methodology & Evaluation

To ensure academic rigor, we implemented an automated evaluation framework inspired by Google's *Purpose-Driven Evaluation*. We test against a **Golden Dataset** covering diverse scenarios.

* **Pillar 1: Agent Success & Quality:** Verified by comparing agent-extracted numbers against ground truth using a semantic LLM Judge.
* **Pillar 2: Process & Trajectory:** Verified by a heuristic validator that ensures the agent selects the correct tool (e.g., using `pandas` for math, not text prediction).
* **Pillar 3: Trust & Safety:** Verified by "Negative Tests" to ensure the agent reports "Data Missing" rather than hallucinating metrics for non-existent companies.

---

## ✨ Features & Capabilities

This project implements four core components required by the course curriculum, alongside several advanced bonus features.

### ✅ Core Components (Course Requirements)
1.  **Retrieval Augmented Generation (RAG):**
    * **Implementation:** Queries a local ChromaDB vector store using OpenAI embeddings to retrieve source-referenced business summaries and news.
2.  **Data Analysis & Code Execution:**
    * **Implementation:** The agent autonomously writes and executes `pandas` code to analyze a structured CSV dataset, calculating metrics like volatility distributions.
3.  **Multi-step Agent Pipeline:**
    * **Implementation:** Utilizes a Planner-Executor model where the `CodeAgent` breaks down complex user prompts into logical steps (e.g., Load Data → Check News → Plot Comparison). A second LLM acts as "Compliance Officer" and reviews output from CodeAgent.
4.  **Image Generation Integration:**
    * **Implementation:** Integrated via function calling to DALL-E 3 for generating illustrative visuals of abstract market concepts.

### 🌟 Bonus & Advanced Features
* **Scientific Evaluation Pipeline:** A custom "LLM-as-a-Judge" framework (based on Google's *Purpose-Driven Evaluation*) to audit agent performance across Quality, Process, and Safety pillars.
* **Chain-of-Thought Visualization:** Full transparency in the UI, displaying the agent's internal reasoning traces (Thoughts, Tool Calls, Observations) in real-time.
* **Conversation Memory:** Persistent session state handling in Streamlit, allowing for multi-turn conversations.
* **Robustness & Safety Guardrails:** Strict system prompts preventing financial advice and hallucination (e.g., "The Golden Rule" of dual-sourcing data) and 
* **Self-Correction Loop:** An automated evaluator checks the agent's final output before showing it to the user; if it fails compliance, the agent is forced to rewrite it.