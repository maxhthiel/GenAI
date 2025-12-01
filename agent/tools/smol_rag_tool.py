from smolagents import Tool
import chromadb
from openai import OpenAI
import os

class RAGQueryTool(Tool):

    name = "rag_query"
    description = "Query the ChromaDB vectorstore using OpenAI embeddings."
    inputs = {
        "question": {
            "type": "string",
            "description": "The question to search in the vector DB"
        }
    }
    output_type = "string"   # Pflicht!

    def __init__(self, chroma_path: str, collection_name: str = "nasdaq"):
        super().__init__()
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

    def forward(self, question: str):
        query_emb = self.embed(question)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=5
        )

        docs = results["documents"][0]
        context = "\n\n".join(docs)

        return context
