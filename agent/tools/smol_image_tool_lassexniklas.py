# agent/tools/smol_image_tool_lasse.py
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from smolagents import Tool

class ImageGenerationTool(Tool):
    name = "image_generation_tool"
    description = "Generates images from news summaries for a given company in the CSV."

    inputs = {
        "company_name": {
            "type": "string",
            "description": "Name of the company to generate an image for."
        }
    }
    output_type = "string"

    def __init__(self, csv_path: str = "./data/nasdaq_100_final_for_RAG.csv"):
        super().__init__()
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.csv_path = Path(csv_path)

    def generate_image_prompt(self, article_summary: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful research assistant. 
                    You will be presented with the summary of a news article. 
                    Your task is to generate a highly specific and visual prompt based on this summary. 
                    Generate only the prompt itself, do not include greetings or commentary. 
                    Keep it concise (max 70 tokens) and avoid real people, trademarks, companies, or geopolitical events."""
                },
                {
                    "role": "user",
                    "content": f"Article summary:\n{article_summary}\n\nGenerate an image prompt:"
                }
            ]
        )
        return response.choices[0].message.content

    def generate_image(self, prompt: str) -> str:
        result = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024"
        )
        return result.data[0].url

    def forward(self, company_name: str):
        df = pd.read_csv(self.csv_path)

        row_idx = df.index[df['Company'] == company_name].tolist()
        if not row_idx:
            return f"Company '{company_name}' not found!"
        row_idx = row_idx[0]

        summary = df.at[row_idx, 'News Summary']
        if pd.isna(summary) or summary.strip() == "":
            return f"No summary for '{company_name}' available!"

        prompt = self.generate_image_prompt(summary)
        image_url = self.generate_image(prompt)

        df.at[row_idx, 'Image_URL'] = image_url
        df.to_csv(self.csv_path, index=False)

        return image_url