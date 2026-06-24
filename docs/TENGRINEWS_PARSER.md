# TengriNews Parser

Automatic hourly parser for [tengrinews.kz/news](https://tengrinews.kz/news/).

## What it does

1. Fetches the main news listing page every hour.
2. Collects all article links found on the page.
3. Skips URLs already processed in previous runs (tracked in `seen_urls.txt`).
4. Downloads and parses each new article, extracting:
   - `url` – article source URL
   - `title` – headline
   - `published_at` – publication date/time
   - `main_text` – full article body
5. Saves results to dated JSON files inside `data/tengrinews/`.

## Anti-blocking features

| Technique | Details |
|---|---|
| User-Agent rotation | Picks a random realistic browser UA per request |
| Random delays | Sleeps 1.5–4 s between requests (configurable) |
| Retry with back-off | Up to 3 retries; doubles wait on 5xx; 30 s per attempt on 429 |
| Realistic headers | Accept, Accept-Language, Referer, Cache-Control mimic a real browser |
| Persistent session | `requests.Session` reuses TCP connections |
| Random crawl order | Article list is shuffled before fetching |

## Output format

Each run produces a file `data/tengrinews/tengrinews_YYYY-MM-DD_HH-MM.json`:

```json
[
  {
    "url": "https://tengrinews.kz/news/example-123456/",
    "title": "Заголовок новости",
    "published_at": "2026-04-27T10:30:00",
    "main_text": "Полный текст статьи..."
  }
]
```

Previously seen URLs are stored in `data/tengrinews/seen_urls.txt` – one URL per line.

## Quickstart

**Важно:** в команде `python -c "..."` строки внутри кавычек не должны иметь отступов — иначе Python вернёт `IndentationError`.

```bash
# Один раз (тестовый прогон)
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from earthquakes_parser.parser.tengrinews_parser import TengriNewsParser
articles = TengriNewsParser().run()
for a in articles[:3]:
    print(a.title, '|', a.published_at)
"

# Каждый час (бесконечный цикл), Ctrl+C для остановки
python -m earthquakes_parser.parser.tengrinews_scheduler

# В фоне (терминал свободен)
nohup python -m earthquakes_parser.parser.tengrinews_scheduler > data/tengrinews/scheduler.log 2>&1 &
echo "PID: $!"

# Следить за логами фонового процесса
tail -f data/tengrinews/scheduler.log
```

## Configuration

Pass keyword arguments to `TengriNewsParser`:

```python
from earthquakes_parser.parser.tengrinews_parser import TengriNewsParser

parser = TengriNewsParser(
    output_dir="data/tengrinews",   # where to write JSON files
    min_delay=1.5,                  # minimum seconds between requests
    max_delay=4.0,                  # maximum seconds between requests
    max_retries=3,                  # retry attempts per failed request
    timeout=20,                     # HTTP timeout in seconds
)
parser.run()
```

To change the schedule interval:

```python
from earthquakes_parser.parser.tengrinews_scheduler import run_scheduler

run_scheduler(output_dir="data/tengrinews", interval=1800)  # every 30 min
```

## Module overview

| File | Purpose |
|---|---|
| `tengrinews_parser.py` | Core parser: fetches listing, parses articles, persists JSON |
| `tengrinews_scheduler.py` | Infinite loop that calls the parser every hour |

## CSS selectors

The parser tries multiple selectors in order and uses the first match.
If the site redesigns, update the `_TITLE_SELECTORS`, `_DATE_SELECTORS`,
`_BODY_SELECTORS`, and `_LINK_SELECTORS` lists at the top of `tengrinews_parser.py`.
