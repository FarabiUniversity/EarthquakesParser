# SchemaExtractor – GPT-Based Schema Generation

## Overview

`SchemaExtractor` analyzes HTML structure using GPT to generate CSS selector schemas. It handles token counting, HTML truncation, and JSON parsing from LLM responses.

## Architecture

```
SchemaExtractor
├── extract_schema()      # Main workflow
├── count_tokens()        # Token counting API
├── _build_prompt()       # Prompt construction
├── _truncate_html()      # HTML size management
└── _extract_json()       # JSON parsing
```

## Responsibilities

* Send HTML to GPT for analysis
* Generate CSS selectors for content extraction
* Validate page relevance (earthquake-related)
* Handle token limits and HTML truncation
* Parse structured JSON from LLM responses

## Quick Start

```python
from earthquakes_parser.parser import SchemaExtractor

extractor = SchemaExtractor(
    openai_base_url="http://192.168.8.22:9999/v1",
    openai_api_key="YOUR_API_KEY"  # pragma: allowlist secret
)

schema = extractor.extract_schema(
    html="<html>...</html>",
    title="Earthquake Report",
    domain="example.com"
)

if schema:
    print(f"Selectors: {schema.main_text_selectors}")
    print(f"Valid: {schema.is_valid}")
```

## API Reference

### Constructor

```python
SchemaExtractor(
    openai_base_url: str = "http://192.168.8.22:9999/v1",
    openai_api_key: str = "YOUR_API_KEY",  # pragma: allowlist secret
    model: str = "gpt-4",
    max_tokens: int = 80000
)
```

**Parameters:**
* `openai_base_url`: GPT API endpoint (supports OpenAI-compatible APIs)
* `openai_api_key`: Authentication key
* `model`: Model name (e.g., `"gpt-4"`, `"gpt-3.5-turbo"`)
* `max_tokens`: Maximum prompt size (default: 80000 ≈ 60k words)

**Supported APIs:**
* OpenAI
* LM Studio (local)
* Ollama (local)
* Any OpenAI-compatible endpoint

---

### `extract_schema()`

Generate schema from HTML using GPT.

```python
extractor.extract_schema(
    html: str,
    title: str,
    domain: str
) -> Optional[PageSchema]
```

**Args:**
* `html`: Full HTML content (already downloaded)
* `title`: Page title for context
* `domain`: Domain name for schema storage

**Returns:** `PageSchema` object or `None` on failure

**Workflow:**
1. Truncate HTML if exceeds token limit
2. Build GPT prompt with instructions
3. Call GPT API
4. Parse JSON response
5. Create `PageSchema` object

**Example:**

```python
html = "<html><article><p>Earthquake occurred...</p></article></html>"

schema = extractor.extract_schema(
    html=html,
    title="M 7.0 Earthquake",
    domain="usgs.gov"
)

# schema.main_text_selectors = ["article p"]
# schema.date_selector = "time.published"
# schema.is_valid = True
```

---

### `count_tokens()`

Count tokens in text using tokenizer API.

```python
extractor.count_tokens(text: str) -> int
```

**Args:**
* `text`: String to count tokens for

**Returns:** Token count (int), or `0` on error

**Example:**

```python
text = "This is a long document..."
tokens = extractor.count_tokens(text)
print(f"Tokens: {tokens}")
```

**API Endpoint:**
```
POST http://192.168.8.22:9999/extras/tokenize/count
Body: {"input": "text here"}
Response: {"count": 1234}
```

---

### Internal Methods

#### `_build_prompt()`

Constructs GPT prompt with HTML and instructions.

```python
prompt = extractor._build_prompt(html, title)
```

**Prompt Structure:**
1. Task description
2. JSON schema format
3. Selector requirements
4. HTML content

**Output Format:**
```json
{
  "schema": {
    "main_text": ["CSS selector 1", "CSS selector 2"],
    "date": "CSS selector for date"
  },
  "is_valid": true
}
```

#### `_truncate_html()`

Reduces HTML size to fit token limit.

```python
truncated = extractor._truncate_html(html, title)
```

**Algorithm:**
```python
token_count = count_tokens(html)
if token_count > max_tokens:
    ratio = max_tokens / token_count * 0.9
    html = html[:int(len(html) * ratio)]
```

#### `_extract_json()`

Parses JSON from markdown-wrapped GPT response.

```python
result = extractor._extract_json(response_text)
```

**Input:**
```
Here's the schema:
```json
{"schema": {...}, "is_valid": true}
```
```

**Output:**
```python
{"schema": {...}, "is_valid": True}
```

## GPT Prompt Design

### Instructions

```
You are given HTML content of a webpage titled "{title}".

Analyze the structure and return:
{
  "schema": {
    "main_text": ["CSS selectors for main content"],
    "date": "CSS selector for publication date"
  },
  "is_valid": true if about earthquakes, false otherwise
}
```

### Selector Requirements

**main_text:**
* Target paragraphs and article sections
* Avoid navigation, footers, sidebars
* Isolate meaningful content blocks

**date:**
* Prefer metadata elements
* Look for `<time>`, `<meta>` tags
* Match publication/updated dates

### Validation

GPT determines `is_valid` based on:
* Keywords: earthquake, seismic, tremor, magnitude
* Content context and topic
* Domain relevance

## Response Handling

### Successful Response

```json
{
  "schema": {
    "main_text": [".article__content p", ".main-text"],
    "date": "time.published"
  },
  "is_valid": true
}
```

Converts to:
```python
PageSchema(
    domain="example.com",
    main_text_selectors=[".article__content p", ".main-text"],
    date_selector="time.published",
    is_valid=True
)
```

### Failed Response

Returns `None` if:
* JSON parsing fails
* API timeout
* Invalid response format
* Network error

## Token Management

### Why It Matters

Large HTML files can exceed model context limits:
* GPT-4: 128k tokens
* GPT-3.5: 16k tokens
* Local models: varies

### Truncation Strategy

```python
# Example: 150k tokens → 80k limit
html_length = 200000 chars
token_count = 150000 tokens

ratio = 80000 / 150000 * 0.9 = 0.48
truncated_length = 200000 * 0.48 = 96000 chars
```

**Safety margin (0.9):**
Accounts for prompt instructions overhead

## Error Handling

### Network Errors

```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"❌ Error calling GPT: {e}")
    return None
```

### JSON Parsing Errors

```python
try:
    return json.loads(match.group(1))
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
    return None
```

### Token Counting Errors

```python
try:
    return response.json().get("count", 0)
except Exception as e:
    print(f"⚠️ Error counting tokens: {e}")
    return 0  # Safe fallback
```

## Performance Tips

### 1. Cache Token Counts

```python
# Avoid re-counting same HTML
token_cache = {}
if html not in token_cache:
    token_cache[html] = extractor.count_tokens(html)
```

### 2. Adjust Temperature

```python
# Lower temperature = more consistent selectors
response = client.chat.completions.create(
    temperature=0.2  # More deterministic
)
```

### 3. Retry Logic

```python
max_retries = 3
for attempt in range(max_retries):
    schema = extractor.extract_schema(html, title, domain)
    if schema:
        break
```

## Integration

### With ParserManager

```python
# Automatic schema extraction
if not schema_manager.exists(domain):
    schema = schema_extractor.extract_schema(html, title, domain)
    schema_manager.save(schema)
```

### With SchemaManager

```python
# Extract and save pipeline
schema = extractor.extract_schema(html, title, domain)
if schema and schema.is_valid:
    schema_id = manager.save(schema)
```

## Benefits

✅ **Automatic Schema Generation** – No manual CSS selector writing
✅ **Page Validation** – Filters irrelevant content
✅ **Token Management** – Handles large HTML files
✅ **Flexible API Support** – Works with any OpenAI-compatible API
✅ **Robust Parsing** – Handles various GPT response formats

## Limitations

⚠️ **API Dependency** – Requires GPT access
⚠️ **Cost** – API calls have usage fees
⚠️ **Speed** – Slower than pre-defined schemas
⚠️ **Accuracy** – GPT may generate incorrect selectors

## See Also

* `PARSER_MANAGER.md` – Schema usage
* `SCHEMA_MANAGER.md` – Schema persistence
* `DATA_EXTRACTOR.md` – Selector application
* `MODELS.md` – PageSchema structure
