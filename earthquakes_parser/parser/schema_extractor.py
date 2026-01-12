"""Schema extraction from HTML using GPT."""

import json
import re
from typing import Optional

import requests
from openai import OpenAI

from .models import PageSchema


class SchemaExtractor:
    """Extracts page schemas using GPT."""

    def __init__(
        self,
        openai_base_url: str = "http://192.168.8.22:9999/v1",
        openai_api_key: str = "api-key",
        model: str = "gpt-4",
        max_tokens: int = 80000,
    ):
        """Initialize schema extractor.

        Args:
            openai_base_url: Base URL for OpenAI-compatible API.
            openai_api_key: API key.
            model: Model name to use.
            max_tokens: Maximum tokens for prompt.
        """
        self.client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.tokenizer_url = "http://192.168.8.22:9999/extras/tokenize/count"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count or 0 if failed.
        """
        try:
            response = requests.post(
                self.tokenizer_url,
                headers={"Content-Type": "application/json"},
                json={"input": text},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("count", 0)
        except Exception as e:
            print(f"⚠️ Error counting tokens: {e}")
            return 0

    def _build_prompt(self, html: str, title: str) -> str:
        """Build GPT prompt for schema extraction.

        Args:
            html: HTML content.
            title: Page title.

        Returns:
            Formatted prompt.
        """
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
{{...}}
```

Here is the HTML content:
{html}
"""

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from GPT response.

        Args:
            text: GPT response text.

        Returns:
            Parsed JSON dict or None.
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
        return None

    def _truncate_html(self, html: str, title: str) -> str:
        """Truncate HTML to fit token limit.

        Args:
            html: HTML content.
            title: Page title.

        Returns:
            Truncated HTML.
        """
        prompt = self._build_prompt(html, title)
        token_count = self.count_tokens(prompt)

        if token_count > self.max_tokens:
            print(f"⚠️ Prompt contains {token_count} tokens. Truncating HTML.")
            max_chars = int(len(html) * (self.max_tokens / token_count * 0.9))
            html = html[:max_chars]

        return html

    def extract_schema(
        self, html: str, title: str, domain: str
    ) -> Optional[PageSchema]:
        """Extract schema from HTML using GPT.

        Args:
            html: HTML content (already downloaded).
            title: Page title.
            domain: Domain name.

        Returns:
            PageSchema object or None if failed.
        """
        # Truncate if needed
        html = self._truncate_html(html, title)

        # Build prompt
        prompt = self._build_prompt(html, title)

        # Call GPT
        try:
            print(f"🔍 Analyzing: {title}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts schema from HTML.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            result_text = response.choices[0].message.content
            result = self._extract_json(result_text)

            if not result:
                print(f"❌ Failed to extract JSON from response")
                return None

            print(f"📄 GPT Response:\n{result}")

            # Parse schema
            schema_data = result.get("schema", {})
            is_valid = result.get("is_valid", False)

            return PageSchema(
                domain=domain,
                main_text_selectors=schema_data.get("main_text", []),
                date_selector=schema_data.get("date"),
                is_valid=is_valid,
            )

        except Exception as e:
            print(f"❌ Error calling GPT: {e}")
            return None