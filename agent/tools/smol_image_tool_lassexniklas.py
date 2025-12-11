import os
from typing import List
from smolagents import Tool
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
env_path = Path(".") / ".env"
load_dotenv(env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def generate_image_prompt(article_summary: str) -> str:
    """Generates a prompt based on the article summary."""
    prompt = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a helpful research assistant.
            You will be presented with the summary of a news article.
            Your task is to generate a highly specific and visual prompt based on this summary.
            Generate only the prompt itself, no greetings or commentary.
            Keep it concise (max 70 tokens).
            Avoid real people, trademarks, companies, or geopolitical events."""},
            {"role": "user", "content": f"Article summary:\n{article_summary}\n\nGenerate an image prompt:"}
        ]
    ).choices[0].message.content
    return prompt

def generate_image(prompt: str) -> str:
    """Generates an image URL using DALL-E."""
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    url = result.data[0].url
    if url is None:
        raise RuntimeError("Image API returned no image URL!")
    return url

# Optional: Bewertung der Bilder (kann bleiben)
def embed_text(text: str) -> np.ndarray:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    return np.array(emb)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - cosine(vec1, vec2)

# ------------------------------------------------------------
# Pipeline: Generiert mehrere Bilder, gibt besten Link zurück
# ------------------------------------------------------------
def image_gen_pipeline(article_summary: str, num_images: int = 2) -> str:
    prompt = generate_image_prompt(article_summary)
    images = []
    ratings = []

    for i in range(num_images):
        image_url = generate_image(prompt)
        images.append(image_url)
        # Optional: einfache Bewertung (z.B. zufällig oder Embeddings)
        ratings.append(1)  # placeholder: gleiche Bewertung

    # Bestes Bild auswählen (hier einfach das erste, da Bewertung placeholder)
    best_index = ratings.index(max(ratings))
    best_image_url = images[best_index]

    return best_image_url

# ------------------------------------------------------------
# Tool Class
# ------------------------------------------------------------
class ImageGenerationTool(Tool):
    name = "image_generation_tool"
    description = "Generates an image based on a news article summary"
    
    inputs = {
        "article_summary": {
            "type": "string",
            "description": "The news article summary to generate an image for."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, article_summary: str) -> str:
        # Gibt direkt den Link zurück
        best_image_url = image_gen_pipeline(article_summary)
        return best_image_url


# ------------------------------------------------------------
# Beispiel für Nutzung
# ------------------------------------------------------------
if __name__ == "__main__":
    tool = ImageGenerationTool()
    summary = "AstraZeneca announces a new breakthrough in cancer research."
    url = tool.forward(summary)
    print("Generated image URL:", url)
