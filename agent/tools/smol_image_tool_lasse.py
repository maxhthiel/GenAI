import os
from typing import List
from smolagents import Tool
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import base64
import requests

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
env_path = Path(".") / ".env"
load_dotenv(env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------
# Helper functions (used to build the pipeline)
# ------------------------------------------------------------

def generate_image_prompt(article_summary: str) -> str:
    """Generates a prompt based on the article summary."""
    prompt = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a helpful research assistant.
            You will be presented with the summary of a news article.
            Your task is to generate a highly speficic and visual prompt based on this summary.
            Generate only the prompt itself, further greetings or comments are not needed.
            Keep the prompt concise, it should not exceed 70 tokens.
            Make sure to avoid the names of real people, trademarks, companies or geopolitical events.
            This prompt will later be used to generate images via another LLM."""},
            {"role": "user", "content": f"Article summary:\n{article_summary}\n\nGenerate an image prompt:"}
        ]
    ).choices[0].message.content
    return prompt


def generate_image(prompt: str) -> str:
    """Generates image URL using DALL-E."""
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    url = result.data[0].url
    if url is None:
        raise RuntimeError("Image API returned no image URL!")
    return url


def describe_image(image_url: str) -> str:
    """Uses Vision model to describe an image via URL."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image. Focus on the key features. The description should be concise and not exceed 100 tokens."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        # Falls Liste zurückgegeben wird
        return "".join([c.get("text", "") for c in content])
    return content


def embed_text(text: str) -> np.ndarray:
    """Returns an embedding vector for a given text."""
    emb = client.embeddings.create(
        model="text-embedding-3-small",   # Simple embedding model
        input=text
    ).data[0].embedding

    return np.array(emb)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Returns cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def rate_image_embedding(prompt: str, image_url: str) -> float:
    """
    Rates how well an image matches a prompt using:
      1) Image URL -> Vision Description
      2) Prompt & Description -> embeddings
      3) Cosine similarity -> score in range 0–10
    """
    print("Describing image...")
    image_desc = describe_image(image_url)  # jetzt korrekt URL übergeben
    print("Image description:", image_desc)
    
    # Compute embeddings
    text_emb = embed_text(prompt)
    desc_emb = embed_text(image_desc)
    
    # Cosine similarity
    sim = cosine_similarity(text_emb, desc_emb)
    score = max(0, min(10, sim * 10))
    return score


def find_best_image(images: List[bytes], ratings: List[float]) -> bytes:
    """Select the best image according to its rating."""
    best_index = max(range(len(ratings)), key=lambda i: ratings[i])
    return images[best_index]


# ------------------------------------------------------------
# Main Pipeline (consisting of helper functions)
# ------------------------------------------------------------

def image_gen_pipeline(article_summary: str) -> bytes:
    """
    Full pipeline (based on the structure given in our script):
      1) Generate a prompt based on the article summary
      2) Generate several images and rate them
      3) Return best image (as bytes)
    """

    # Step 1: Build image prompt
    prompt = generate_image_prompt(article_summary)
    print("Prompt:", prompt) ###########################

    images = []
    ratings = []

    # Step 2: Generate & evaluate 2 images
    for i in range(2):
        print(f"--- Generating image {i} ---")
        image_url = generate_image(prompt)
        print(f"Image {i} URL:", image_url)

        rate = rate_image_embedding(prompt, image_url)
        images.append(image_url)  # optional: runterladen, wenn du Bytes brauchst
        ratings.append(rate)

    # Step 3: Return best image
    best_index = ratings.index(max(ratings))
    best_image_url = images[best_index]
    print(f"Best image is image {best_index} with rating {ratings[best_index]:.2f}")
   
    img_bytes = requests.get(best_image_url).content
    return img_bytes


# ------------------------------------------------------------
# Tool Class for smolagents (follows the pipeline)
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
        best_image_bytes = image_gen_pipeline(article_summary)

        # Base64 string instead of raw bytes (forward needs JSON output)
        return base64.b64encode(best_image_bytes).decode("utf-8")