"""Test Gemini API for earthquake data retrieval."""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", tools="google_search_retrieval"
)

response = model.generate_content("earthquake damage statistics 2024")
print(response.text)
