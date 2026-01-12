# ParserManager – Earthquake Content Parser

## Overview

`ParserManager` orchestrates the complete content extraction pipeline from downloaded HTML files. It coordinates schema extraction, data parsing, and content persistence.

**Key Components:**
* `SchemaManager`: CRUD operations for CSS selector schemas
* `SchemaExtractor`: GPT-based schema generation from HTML
* `DataExtractor`: Content extraction using BeautifulSoup
* `SearchManager`: Status tracking integration
* `SupabaseFileStorage`: HTML file retrieval from storage

## Architecture

```
ParserManager
├── SchemaManager (Database CRUD)
│   ├── get_by_domain()
│   ├── save()
│   └── exists()
├── SchemaExtractor (GPT Integration)
│   ├── extract_schema()
│   ├── count_tokens()
│   └── _truncate_html()
├── DataExtractor (HTML Parsing)
│   ├── extract()
│   ├── _extract_main_text()
│   └── _extract_date()
└── SupabaseFileStorage (HTML Retrieval)
    └── download()
```

## Responsibilities

* Download HTML from Supabase Storage
* Extract or retrieve CSS selector schemas
* Parse HTML content using schemas
* Handle schema re-extraction on failures
* Save parsed content to database
* Track parsing status

## Quick Start

```python
from earthquakes_parser.parser import ParserManager
from earthquakes_parser.storage.supabase import SupabaseDB, SupabaseFileStorage

db = SupabaseDB()
storage = SupabaseFileStorage(bucket_name="html-files")
parser = ParserManager(db, storage)

# Parse all downloaded URLs
stats = parser.parse_downloaded(limit=50)
print(f"✅ Parsed: {stats['successful']}, ❌ Failed: {stats['failed']}")
```

## API Reference

### Constructor

```python
ParserManager(
    db: SupabaseDB,
    file_storage: SupabaseFileStorage,
    openai_base_url: str = "http://192.168.8.22:9999/v1",
    openai_api_key: str = "api-key"
)
```

**Parameters:**
* `db`: Database instance for persistence
* `file_storage`: Storage instance for HTML retrieval
* `openai_base_url`: GPT API endpoint
* `openai_api_key`: API authentication key

---

### `parse_record()`

Parse a single search result record.

```python
parser.parse_record(
    record: dict,
    force_reextract: bool = False
) -> bool
```

**Args:**
* `record`: Dict with keys `id`, `link`, `title`, `html_storage_path`
* `force_reextract`: Force schema re-extraction even if cached

**Returns:** `True` if parsing succeeded, `False` otherwise

**Workflow:**
1. Download HTML from storage
2. Check/extract schema for domain
3. Extract main text and date
4. Re-extract schema if extraction fails
5. Save to `parsed_content` table
6. Update status to `parsed` or `failed`

**Example:**

```python
record = {
    'id': 'uuid-here',
    'link': 'https://example.com/article',
    'title': 'Earthquake Report',
    'html_storage_path': 'path/to/file.html'
}

success = parser.parse_record(record)
```

---

### `parse_downloaded()`

Parse all URLs with downloaded HTML.

```python
parser.parse_downloaded(limit: Optional[int] = 100) -> dict
```

**Args:**
* `limit`: Maximum number of records to process

**Returns:**

```python
{
    'total': int,       # Total processed
    'successful': int,  # Successfully parsed
    'failed': int      # Failed to parse
}
```

**Example:**

```python
stats = parser.parse_downloaded(limit=50)
print(f"Success rate: {stats['successful']/stats['total']*100:.1f}%")
```

---

### `get_statistics()`

Get parsing statistics across all statuses.

```python
parser.get_statistics() -> dict
```

**Returns:**

```python
{
    'total': int,
    'pending': int,
    'downloaded': int,
    'parsed': int,
    'analyzed': int,
    'failed': int
}
```

---

### `update_parsed_content()`

Update existing parsed content.

```python
parser.update_parsed_content(
    parsed_content_id: str,
    main_text: Optional[list] = None,
    date: Optional[str] = None,
    page_schema_id: Optional[str] = None
) -> bool
```

---

### `mark_as()`

Update status of parsed content.

```python
parser.mark_as(parsed_content_id: str, status: str) -> bool
```

## Parsing Logic

### Schema Handling

1. **Check existing schema:**
   ```python
   schema = schema_manager.get_by_domain("example.com")
   ```

2. **If not found → Extract via GPT:**
   ```python
   schema = schema_extractor.extract_schema(html, title, domain)
   schema_id = schema_manager.save(schema)
   ```

3. **Validate page relevance:**
   ```python
   if not schema.is_valid:
       mark_as(id, "failed")
       return False
   ```

### Extraction & Validation

1. **Extract content:**
   ```python
   result = data_extractor.extract(html, schema)
   ```

2. **Validate result:**
   - ✅ **Success:** `main_text` is not empty (date can be None)
   - ⚠️ **Re-extract:** `main_text` is empty → try new schema
   - ❌ **Failed:** Still empty after re-extraction

3. **Save result:**
   ```python
   save_parsed_content(search_result_id, main_text, date, schema_id)
   mark_as(parsed_content_id, "parsed")
   ```

## Status Workflow

```
downloaded → parsing → parsed
                    ↓
                  failed
```

**Status Meanings:**
* `downloaded`: HTML in storage, ready to parse
* `parsed`: Content successfully extracted
* `failed`: `main_text` extraction failed (date is optional)

## Error Handling

### Automatic Recovery

```python
if not result.main_text:
    # Re-extract schema
    schema = schema_extractor.extract_schema(html, title, domain)
    schema_manager.save(schema)
    
    # Try again
    result = data_extractor.extract(html, schema)
    
    if not result.main_text:
        mark_as(id, "failed")
```

### Common Failures

| Issue | Cause | Solution |
|-------|-------|----------|
| No HTML | Missing storage path | Check `html_storage_path` |
| Schema extraction fails | GPT timeout/error | Retry or check API |
| Invalid page | Not earthquake-related | GPT validation → `is_valid=false` |
| Empty extraction | Wrong selectors | Automatic schema re-extraction |

## Benefits

✅ **Schema Caching** – Reuse schemas for same domain  
✅ **Auto Recovery** – Re-extract schema on failure  
✅ **Flexible Validation** – `main_text` required, `date` optional  
✅ **Status Tracking** – Clear pipeline visibility  
✅ **Modular Design** – Each component has single responsibility  

## Performance Tips

1. **Batch Processing:**
   ```python
   # Process in chunks
   parser.parse_downloaded(limit=100)
   ```

2. **Monitor Statistics:**
   ```python
   stats = parser.get_statistics()
   remaining = stats['downloaded']
   ```

3. **Force Re-extraction:**
   ```python
   # For sites with changed structure
   parser.parse_record(record, force_reextract=True)
   ```

## See Also

* `SCHEMA_MANAGER.md` – Schema persistence
* `SCHEMA_EXTRACTOR.md` – GPT integration
* `DATA_EXTRACTOR.md` – HTML parsing
* `MODELS.md` – Data structures
* `SUPABASE_ARCHITECTURE.md` – Overall system