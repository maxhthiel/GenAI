import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os


class RAGTool:
    """
    Ein einfaches RAG-Modul:
    - lädt einen persistierten Chroma-Vectorstore
    - encoded Querries
    - findet relevante Dokumente
    - optional: lässt OpenAI aus dem Kontext Antworten generieren
    """

    def __init__(self, chroma_path: str, collection_name: str = "nasdaq"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # OpenAI client über .env
        self.model = os.getenv("OPENAI_MODEL")
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def query(self, question: str, top_k: int = 5):
        embedding = self.encoder.encode(question).tolist()

        # relevante Dokumente aus Vector DB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        context_blocks = results["documents"][0]
        context = "\n\n".join(context_blocks)

        # LLM-Antwort generieren
        completion = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Du bist ein Experte für Unternehmensdaten."},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}
            ]
        )

        return {
            "context": context_blocks,
            "answer": completion.choices[0].message["content"]
        }
