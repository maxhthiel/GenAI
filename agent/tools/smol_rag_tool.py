from smolagents import Tool
import chromadb
from openai import OpenAI
import os

class RAGQueryTool(Tool):
    name = "rag_query"
    description = "PRIMARY source for financial information. Searches the internal high-quality database for specific company details, sentiment analysis, and historical context. ALWAYS use this BEFORE web search for company questions."
    inputs = {
        "question": {
            "type": "string",
            "description": "The question to search in the vector DB"
        }
    }
    output_type = "string"

    def __init__(self, chroma_path: str, collection_name: str = "nasdaq_docs"):
        super().__init__()
        # Initialisierung
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"

    def embed(self, text: str):
        emb = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return emb.data[0].embedding

    def forward(self, question: str) -> str:
        query_emb = self.embed(question)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        # Sicherstellen, dass wir Text zurückgeben
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            return "\n\n".join(docs)
        return "No relevant documents found."