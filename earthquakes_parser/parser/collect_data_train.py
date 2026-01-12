import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dateutil import parser as date_parser
import os

client = OpenAI(base_url="http://192.168.8.22:9999/v1", api_key="api-key")  # Замените на ваш ключ

# Загрузка CSV-файла
df = pd.read_csv("sandbox/data/web_results.csv")

def extract_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"Ошибка при парсинге JSON: {e}")
    return None


def extract_data_from_html(html, schema):
    soup = BeautifulSoup(html, "html.parser")
    extracted = {}

    main_paths = schema.get("main_text", [])
    main_texts = []
    for path in main_paths:
        elements = soup.select(path)
        for el in elements:
            text = el.get_text(strip=True)
            if text:
                main_texts.append(text)
    extracted["main_text"] = main_texts

    date_path = schema.get("date")

    if date_path:
        date_elements = soup.select(date_path)
        if date_elements:
            date_text = date_elements[0].get_text(strip=True)
            try:
                parsed_date = date_parser.parse(date_text, fuzzy=True).date()
                extracted["date"] = parsed_date.isoformat()  # формат YYYY-MM-DD
            except Exception as e:
                print(f"⚠️ Не удалось распарсить дату: {e}")
                extracted["date"] = date_text or None
        else:
            extracted["date"] = None
    else:
        extracted["date"] = None

    return extracted


def count_tokens(text):
    try:
        response = requests.post(
            "http://192.168.8.22:9999/extras/tokenize/count",
            headers={"Content-Type": "application/json"},
            json={"input": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("count", 0)
    except Exception as e:
        print(f"⚠️ Ошибка при подсчёте токенов: {e}")
        return 0


def fetch_html(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.prettify()
    except Exception as e:
        print(f"❌ Ошибка при загрузке {url}: {e}")
        return None

def build_prompt(html, title):
    return f"""
You are given the full HTML content of a webpage titled "{title}". Your task is to analyze the structure and return a JSON object in the following format:

{{
  "schema": {{
    "main_text": ["CSS-like selectors pointing to the main content blocks, such as paragraphs or article sections. Each selector should isolate a meaningful unit of text, like a paragraph, article body, or section. Avoid selectors that include navigation, footers, sidebars, references, or link lists."],
    "date": "CSS-like selector pointing to the element that contains the publication or last updated date. Prefer metadata or footer elements with clear date formatting."
  }},
  "is_valid": true if the page is about earthquakes or closely related topics, false otherwise
}}

⚠️ Important: Do not return anything except the JSON object wrapped in triple backticks like this:
```json

Here is the HTML content:
{html}
"""

def analyze_html_with_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts schema from HTML."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def main():
    for _, row in df.iterrows():
        query, link, title = row["query"], row["link"], row["title"]
        html = fetch_html(link)
        if html:
            prompt = build_prompt(html, title)
            token_count = count_tokens(prompt)
            if token_count > 100000:
                print(f"⚠️ Промт содержит {token_count} токенов. Обрезаем HTML до лимита.")
                max_chars = len(html) * (80000 / token_count)
                html = html[:int(max_chars)]
                prompt = build_prompt(html, title)

            print(f"\n🔍 Обработка: {title} ({link})")
            result = extract_json(analyze_html_with_openai(prompt))
            print(f"📄 Ответ модели:\n{result}\n{'='*100}")
            extracted = extract_data_from_html(html, result["schema"])
            print(f"\n📌 Извлечённый текст:\n{extracted['main_text']}")
            print(f"📅 Дата на странице: {extracted['date']}")


if __name__ == "__main__":
    main()