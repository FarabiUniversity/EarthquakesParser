# SchemaManager – Page Schema Management

## Overview

`SchemaManager` handles CRUD operations for page schemas in the `page_schemas` database table. Schemas define CSS selectors for extracting content from specific domains.

## Architecture

```
SchemaManager
├── get_by_domain()   # Retrieve schema
├── save()            # Insert or update
├── exists()          # Check existence
└── delete()          # Remove schema
```

## Responsibilities

* Store CSS selector schemas per domain
* Deduplicate schemas (one per domain)
* Parse JSON array fields from database
* Provide simple CRUD interface

## Quick Start

```python
from earthquakes_parser.parser import SchemaManager, PageSchema
from earthquakes_parser.storage.supabase import SupabaseDB

db = SupabaseDB()
manager = SchemaManager(db)

# Check if schema exists
if manager.exists("example.com"):
    schema = manager.get_by_domain("example.com")
    print(schema.main_text_selectors)
```

## API Reference

### Constructor

```python
SchemaManager(db: SupabaseDB)
```

**Parameters:**
* `db`: Database instance for persistence

---

### `get_by_domain()`

Retrieve schema for a specific domain.

```python
manager.get_by_domain(domain: str) -> Optional[PageSchema]
```

**Args:**
* `domain`: Domain name (e.g., `"example.com"`)

**Returns:** `PageSchema` object or `None` if not found

**Example:**

```python
schema = manager.get_by_domain("earthquake.usgs.gov")

if schema:
    print(f"Selectors: {schema.main_text_selectors}")
    print(f"Date: {schema.date_selector}")
    print(f"Valid: {schema.is_valid}")
```

**Important:** Automatically parses JSON array from database:

```python
# Database stores: '[".selector1", ".selector2"]'
# Returns: [".selector1", ".selector2"]  (list)
```

---

### `save()`

Save or update schema (upsert operation).

```python
manager.save(schema: PageSchema) -> Optional[str]
```

**Args:**
* `schema`: `PageSchema` object to persist

**Returns:** Schema ID (UUID string) or `None` on failure

**Behavior:**
* **If schema exists for domain:** Updates existing record
* **If schema doesn't exist:** Inserts new record

**Example:**

```python
schema = PageSchema(
    domain="example.com",
    main_text_selectors=["article p", "div.content"],
    date_selector="time.published",
    is_valid=True
)

schema_id = manager.save(schema)
print(f"Saved with ID: {schema_id}")
```

---

### `exists()`

Check if schema exists for domain.

```python
manager.exists(domain: str) -> bool
```

**Args:**
* `domain`: Domain name

**Returns:** `True` if schema exists, `False` otherwise

**Example:**

```python
if not manager.exists("newsite.com"):
    # Extract schema via GPT
    schema = extract_schema(html, title, "newsite.com")
    manager.save(schema)
```

---

### `delete()`

Remove schema by ID.

```python
manager.delete(schema_id: str) -> bool
```

**Args:**
* `schema_id`: UUID of schema

**Returns:** `True` if deleted successfully

**Note:** Rarely used – schemas are typically updated, not deleted

## Database Schema

### `page_schemas` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid | Primary key |
| `domain` | text | Domain name (unique) |
| `main_text_selectors` | text[] | CSS selectors for content |
| `date_selector` | text | CSS selector for date |
| `is_valid` | boolean | Page relevance flag |
| `created_at` | timestamptz | Creation timestamp |
| `updated_at` | timestamptz | Last update timestamp |

### Example Row

```json
{
  "id": "c6c2b3fd-d73e-43bb-924f-9300fc755922",
  "domain": "tengrinews.kz",
  "main_text_selectors": [".content_main_text p", ".content_main_desc p"],
  "date_selector": "meta[itemprop='datePublished']",
  "is_valid": true,
  "created_at": "2025-11-25T11:29:57.799391+00:00",
  "updated_at": "2025-11-25T11:29:57.799391+00:00"
}
```

## JSON Array Handling

### Problem

PostgreSQL stores arrays correctly, but Supabase Python client sometimes returns them as strings:

```python
# Expected: [".selector1", ".selector2"]
# Actual:   '[".selector1", ".selector2"]'  (string!)
```

### Solution

`get_by_domain()` automatically handles both cases:

```python
main_text_selectors = row["main_text_selectors"]

if isinstance(main_text_selectors, str):
    # Parse JSON string
    import json
    main_text_selectors = json.loads(main_text_selectors)
elif not isinstance(main_text_selectors, list):
    # Fallback to empty list
    main_text_selectors = []
```

## Usage Patterns

### 1. Check → Extract → Save

```python
schema = manager.get_by_domain(domain)

if not schema:
    # Extract via GPT
    schema = schema_extractor.extract_schema(html, title, domain)
    manager.save(schema)
```

### 2. Update Existing Schema

```python
schema = manager.get_by_domain("example.com")
schema.main_text_selectors.append(".new-selector")
manager.save(schema)  # Updates existing
```

### 3. Bulk Check

```python
domains = ["site1.com", "site2.com", "site3.com"]
missing = [d for d in domains if not manager.exists(d)]
print(f"Need schemas for: {missing}")
```

## Error Handling

All methods catch exceptions and return safe defaults:

```python
try:
    schema = manager.get_by_domain(domain)
except Exception as e:
    print(f"❌ Error: {e}")
    return None  # Safe fallback
```

## Benefits

✅ **Automatic Upsert** – No need to check existence before save
✅ **Domain Deduplication** – One schema per domain
✅ **JSON Parsing** – Handles array serialization transparently
✅ **Type Safety** – Returns `PageSchema` dataclass
✅ **Error Resilience** – Graceful failure handling

## Integration

### With ParserManager

```python
# ParserManager uses SchemaManager internally
parser = ParserManager(db, storage)

# Automatically checks schema
result = parser.parse_record(record)
```

### With SchemaExtractor

```python
# Extract and save in one flow
schema = schema_extractor.extract_schema(html, title, domain)
if schema:
    schema_id = schema_manager.save(schema)
```

## See Also

* `PARSER_MANAGER.md` – Main orchestrator
* `SCHEMA_EXTRACTOR.md` – GPT-based schema generation
* `MODELS.md` – PageSchema structure
* `SUPABASE_ARCHITECTURE.md` – Database design
