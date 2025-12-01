# 🤖 Smol-Quant: Autonomous Financial Analyst Agent

## Project Overview
Smol-Quant is a GenAI-powered agent designed to assist financial analysts. Unlike simple chatbots, Smol-Quant can execute Python code to analyze data, retrieve internal company reports via RAG, and generate visualizations.

**Core Features:**
1.  **Retrieval Augmented Generation (RAG):** Queries a local ChromaDB vector store containing Nasdaq 100 financial news and metadata.
2.  **Automated Data Analysis:** Executes Pandas/Python code to perform EDA (Exploratory Data Analysis) on live dataframes.
3.  **Generative Art:** Creates sentiment-based market visualizations using DALL-E 3.
4.  **Multi-Step Reasoning:** Utilizes the `smolagents` framework to plan and execute complex multi-step workflows.

## Architecture
* **Framework:** Hugging Face `smolagents` (CodeAgent)
* **LLM:** OpenAI GPT-4o
* **Vector Database:** ChromaDB (Embeddings: text-embedding-3-small)
* **Interface:** Command Line Interface (CLI) / Streamlit (Optional)

## Setup & Installation

1.  **Clone Repository:**
    ```bash
    git clone <repo-url>
    cd GenAI
    ```

2.  **Install Dependencies:**
    ```bash
    pip install smolagents openai chromadb pandas matplotlib plotly python-dotenv markdownify requests
    ```

3.  **Environment Variables:**
    Create a `.env` file and add your OpenAI Key:
    ```
    OPENAI_API_KEY=sk-proj-...
    ```

4.  **Initialize Database:**
    Populate the vector database with the included CSV data:
    ```bash
    python setup_database.py
    ```

5.  **Run the Agent:**
    ```bash
    python main.py
    ```

## Usage Examples
* **RAG Analysis:** "Nutze die interne Datenbank. Was sind die News zu Nvidia?"
* **Data Analysis:** "Mache eine EDA Analyse des Datensatzes. Gibt es fehlende Werte?"
* **Image Gen:** "Generiere ein Bild, das die Marktstimmung von Tesla visualisiert."