# SearchManager – Earthquake Search Workflow

## Overview

`SearchManager` coordinates the full pipeline for earthquake-related search and data collection. It wraps:

- `BaseSearcher`: abstract interface for search engines
- `GoogleSearcher` / `DDGSearcher`: concrete implementations
- `HTMLDownloade`: modular HTML fetcher supporting both [bs4](https://pypi.org/project/beautifulsoup4/) and [selenium](https://pypi.org/project/selenium/) strategies for dynamic or static page loading
- `SupabaseDB`: database utility for persistence and status tracking

## Architecture
```
SearchManager
├── BaseSearcher (abstract interface)
│   ├── GoogleSearcher
│   └── DDGSearcher
├── HTMLDownloader
│   ├── fetch_with_bs4()
│   └── fetch_with_selenium()
└── SupabaseDB
```

### Responsibilities

- Perform keyword-based searches
- Deduplicate and persist results
- Track status: `pending → downloaded → parsed → analyzed`
- Provide statistics and progress insights

---

## Quick Start
```python
from earthquakes_parser import SearchManager, SupabaseDB, GoogleSearcher

db = SupabaseDB()
searcher = GoogleSearcher()
search_manager = SearchManager(db, searcher)

stats = search_manager.search_and_save(
    keywords=["earthquake", "землетрясение"],
    max_results=5,
    skip_existing=True
)

print(f"New: {stats['new']}, Skipped: {stats['skipped']}")
```

## API Reference

### Constructor
```python
SearchManager(db: SupabaseDB, searcher: Optional[BaseSearcher] = None)
```

- `db`: SupabaseDB instance for persistence
- `searcher`: Optional searcher implementing `BaseSearcher` (defaults to `GoogleSearcher` with env config)

### `search_and_save()`

Search for keywords and save results to database.
```python
search_manager.search_and_save(
    keywords: List[str],
    max_results: int = 5,
    site_filter: Optional[str] = None,
    skip_existing: bool = True
) -> dict
```

**Returns:**
```python
{
    'searched': int,
    'found': int,
    'new': int,
    'skipped': int
}
```

### `get_urls()`

Get URLs with status.
```python
search_manager.get_urls(status: str = "pending", limit: int = 100) -> List[dict]
```

### `mark_as()`

Mark a result as downloaded.
```python
search_manager.mark_as_downloaded(
    search_result_id: str,
    status: str
) -> bool
```

Вот документация в том же стиле для метода `download_html()`:

---

### `download_html()`

Download HTML content for pending URLs and upload to Supabase storage.

```python
search_manager.download_html(storage, fetch_with="selenium", limit=50) -> dict
```

**Args:**
- `storage`: Instance of `SupabaseFileStorage` used for uploading HTML files.
- `fetch_with`: HTML fetch strategy — `"bs4"` for static pages or `"selenium"` for dynamic content.
- `limit`: Maximum number of pending URLs to process.

**Returns:**
```python
{
    'downloaded': int,  # Successfully fetched and uploaded
    'failed': int       # Failed to fetch or upload
}
```

### `get_statistics()`

Get status breakdown.
```python
search_manager.get_statistics() -> dict
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

### `search_with_keywords_file()`

Search using keywords from a file.
```python
search_manager.search_with_keywords_file(
    keywords_file: str,
    max_results: int = 5,
    site_filter: Optional[str] = None,
    skip_existing: bool = True
) -> dict
```

## Status Workflow
```
pending → downloaded
           ↓
         failed
```

## Benefits

✅ **Deduplication**  
Avoids duplicate URLs using `skip_existing`.

✅ **Status Tracking**  
Each result moves through a clear pipeline.

✅ **Statistics**  
Track progress and completion rate.

✅ **Modular Design**  
Clean separation of logic:
```
SearchManager:         Orchestrates full pipeline and business logic
BaseSearcher:          Abstract interface for search engines
GoogleSearcher/DDGSearcher:  Concrete search implementations
HTMLDownloader:        Modular HTML fetcher (bs4 or selenium)
SupabaseDB:            Persistence and status tracking
SupabaseFileStorage:   HTML file upload and linking
```

## See Also

- `SUPABASE_ARCHITECTURE.md` – Overall architecture
- `SUPABASE_USAGE.md` – Supabase utilities guide
- `scripts/test_search_manager.py` – Test script