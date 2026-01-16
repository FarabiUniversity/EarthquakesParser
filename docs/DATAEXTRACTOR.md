# DataExtractor – HTML Content Extraction

## Overview

`DataExtractor` applies CSS selector schemas to HTML using BeautifulSoup to extract main text content and publication dates. It validates extraction results and handles parsing errors gracefully.

## Architecture

```
DataExtractor
├── extract()              # Main extraction
├── _extract_main_text()   # Text content
├── _extract_date()        # Date parsing
└── validate_extraction()  # Result validation
```

## Responsibilities

* Parse HTML with BeautifulSoup
* Apply CSS selectors from schemas
* Extract and clean text content
* Parse publication dates intelligently
* Validate extraction completeness

## Quick Start

```python
from earthquakes_parser.parser import DataExtractor, PageSchema

extractor = DataExtractor()

schema = PageSchema(
    domain="example.com",
    main_text_selectors=["article p", ".content"],
    date_selector="time.published",
    is_valid=True
)

result = extractor.extract(html, schema)

if result.success:
    print(f"Paragraphs: {len(result.main_text)}")
    print(f"Date: {result.date}")
```

## API Reference

### `extract()`

Extract content from HTML using schema.

```python
extractor.extract(
    html: str,
    schema: PageSchema
) -> ExtractionResult
```

**Args:**
* `html`: HTML content as string
* `schema`: `PageSchema` with CSS selectors

**Returns:** `ExtractionResult` with extracted data

**Success Criteria:**
* `success = True` if `main_text` OR `date` found
* `success = False` if both empty

**Example:**

```python
html = """
<article>
  <p>Earthquake of magnitude 7.0 occurred...</p>
  <p>The epicenter was located...</p>
  <time>2024-11-12</time>
</article>
"""

result = extractor.extract(html, schema)

# result.main_text = ["Earthquake of magnitude...", "The epicenter..."]
# result.date = "2024-11-12"
# result.success = True
```

---

### `_extract_main_text()`

Extract text paragraphs using CSS selectors.

```python
texts = extractor._extract_main_text(
    soup: BeautifulSoup,
    selectors: List[str]
) -> List[str]
```

**Args:**
* `soup`: BeautifulSoup parsed HTML
* `selectors`: List of CSS selectors

**Returns:** List of text strings

**Behavior:**
* Applies each selector sequentially
* Extracts text with `strip=True` (removes whitespace)
* Filters empty strings
* Handles selector errors gracefully

**Example:**

```python
selectors = ["article p", "div.content"]

# HTML:
# <article><p>Text 1</p></article>
# <div class="content">Text 2</div>

texts = extractor._extract_main_text(soup, selectors)
# ["Text 1", "Text 2"]
```

**Error Handling:**

```python
for selector in selectors:
    try:
        elements = soup.select(selector)
        # Extract text...
    except Exception as e:
        print(f"⚠️ Error with selector '{selector}': {e}")
        # Continue with next selector
```

---

### `_extract_date()`

Extract and parse publication date.

```python
date = extractor._extract_date(
    soup: BeautifulSoup,
    selector: Optional[str]
) -> Optional[str]
```

**Args:**
* `soup`: BeautifulSoup parsed HTML
* `selector`: CSS selector for date element

**Returns:** ISO date string (`"YYYY-MM-DD"`) or `None`

**Date Parsing:**

Uses `dateutil.parser.parse()` with `fuzzy=True`:

```python
# Input variations:
"November 12, 2024"      → "2024-11-12"
"12.11.2024"             → "2024-11-12"
"2024-11-12T10:30:00Z"   → "2024-11-12"
"Published on 12 Nov"    → "2024-11-12" (current year)
```

**Example:**

```python
# HTML: <time class="published">November 12, 2024</time>
selector = "time.published"

date = extractor._extract_date(soup, selector)
# "2024-11-12"
```

**Error Handling:**

```python
try:
    parsed_date = date_parser.parse(date_text, fuzzy=True).date()
    return parsed_date.isoformat()
except Exception as e:
    print(f"⚠️ Failed to parse date: {e}")
    return None  # Date extraction is optional
```

---

### `validate_extraction()`

Check if extraction needs re-attempt.

```python
needs_reextract = extractor.validate_extraction(
    result: ExtractionResult
) -> bool
```

**Args:**
* `result`: `ExtractionResult` to validate

**Returns:**
* `True` = both fields empty, re-extract schema
* `False` = at least one field populated, extraction OK

**Logic:**

```python
return not result.main_text and result.date is None
```

**Examples:**

| main_text | date | validate_extraction() | Action |
|-----------|------|-----------------------|--------|
| `[]` | `None` | `True` | Re-extract schema |
| `["text"]` | `None` | `False` | Save (date optional) |
| `[]` | `"2024-11-12"` | `False` | Save (unusual but valid) |
| `["text"]` | `"2024-11-12"` | `False` | Save (ideal) |

**Usage in ParserManager:**

```python
result = extractor.extract(html, schema)

if extractor.validate_extraction(result):
    # Both empty → try new schema
    schema = schema_extractor.extract_schema(...)
    result = extractor.extract(html, schema)
```

## Extraction Workflow

### 1. Parse HTML

```python
soup = BeautifulSoup(html, "html.parser")
```

### 2. Extract Main Text

```python
main_texts = []
for selector in schema.main_text_selectors:
    elements = soup.select(selector)
    for el in elements:
        text = el.get_text(strip=True)
        if text:
            main_texts.append(text)
```

### 3. Extract Date

```python
if schema.date_selector:
    elements = soup.select(schema.date_selector)
    if elements:
        date_text = elements[0].get_text(strip=True)
        date = date_parser.parse(date_text, fuzzy=True).date()
```

### 4. Return Result

```python
return ExtractionResult(
    main_text=main_texts,
    date=date,
    success=bool(main_texts) or date is not None
)
```

## CSS Selector Examples

### Common Patterns

```python
# Articles
"article p"
"div.article-body p"
".content__text p"

# News sites
".story-content p"
".article-text"
"#main-content p"

# Metadata
"meta[itemprop='datePublished']"
"time.published"
".article-date"
```

### Selector Best Practices

✅ **Use specific classes:**
```python
".article-content p"  # Good
"p"                   # Too broad
```

✅ **Target content containers:**
```python
"article .body p"     # Good
"article p, aside p"  # Includes sidebar
```

✅ **Avoid navigation:**
```python
".content p"          # Good
"nav p, .content p"   # Includes menu
```

## Error Scenarios

### Invalid Selectors

```python
# Malformed selector
selector = "article[unclosed"

# Handled gracefully:
try:
    elements = soup.select(selector)
except Exception as e:
    print(f"⚠️ Error: {e}")
    # Continue with next selector
```

### No Matches

```python
# Selector doesn't match any elements
selector = ".nonexistent-class"

elements = soup.select(selector)
# elements = []  (empty list, not error)
```

### Unparseable Dates

```python
date_text = "Invalid date format XYZ"

try:
    parsed = date_parser.parse(date_text, fuzzy=True)
except:
    return None  # Date optional
```

## Performance Tips

### 1. Efficient Selectors

```python
# Fast (single class)
".article-text p"

# Slower (descendant + attribute)
"div[data-component='article'] > section p"
```

### 2. Selector Order

```python
# Most specific first
main_text_selectors = [
    "article.main-content p",  # Try specific first
    ".content p",              # Fallback
    "p"                        # Last resort
]
```

### 3. Early Exit

```python
# Stop if enough content found
if len(main_texts) > 100:
    break  # Sufficient content
```

## Integration

### With ParserManager

```python
# Automatic extraction
schema = schema_manager.get_by_domain(domain)
result = data_extractor.extract(html, schema)

if not result.main_text:
    # Re-extract schema
    schema = schema_extractor.extract_schema(...)
    result = data_extractor.extract(html, schema)
```

### With SchemaExtractor

```python
# Test schema immediately
schema = schema_extractor.extract_schema(html, title, domain)
result = data_extractor.extract(html, schema)

if result.success:
    schema_manager.save(schema)
```

## Benefits

✅ **Clean Text Extraction** – `strip=True` removes whitespace
✅ **Flexible Date Parsing** – Handles various formats
✅ **Error Resilience** – Invalid selectors don't crash
✅ **Optional Date** – Content extraction succeeds without date
✅ **Validation Logic** – Clear success criteria

## Limitations

⚠️ **JavaScript Content** – Only parses static HTML
⚠️ **Dynamic Loading** – Can't handle lazy-loaded content
⚠️ **Attribute Values** – Doesn't extract meta content, only visible text

## Common Issues

### Issue: Empty Extraction

**Cause:** Wrong selectors

**Solution:**
```python
# Test selectors in browser DevTools first
# Right-click element → Inspect → Copy selector
```

### Issue: Garbled Text

**Cause:** Wrong encoding

**Solution:**
```python
html = html.encode('latin1').decode('utf-8')
soup = BeautifulSoup(html, 'html.parser')
```

### Issue: Date Not Found

**Cause:** Date in non-text element

**Solution:**
```python
# Try meta tags
"meta[property='article:published_time']"
"meta[name='date']"
```

## See Also

* `PARSER_MANAGER.md` – Extraction orchestration
* `SCHEMA_EXTRACTOR.md` – Schema generation
* `SCHEMA_MANAGER.md` – Schema storage
* `MODELS.md` – ExtractionResult structure
