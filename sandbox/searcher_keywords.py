"""Search for earthquake data using keywords from a file."""

import sys

from dotenv import load_dotenv

from earthquakes_parser.search import GoogleSearcher, SearchManager
from earthquakes_parser.storage.supabase import SupabaseDB

load_dotenv()


def search_from_keywords_file(keywords_file):
    """Read keywords from file and perform search, saving results to DB."""
    with open(keywords_file, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    print(f"Загружено {len(keywords)} ключевых слов из {keywords_file}")

    # Поиск
    db = SupabaseDB()
    searcher = GoogleSearcher()
    search_manager = SearchManager(db, searcher)

    stats = search_manager.search_and_save(
        keywords=keywords,
        max_results=2,
        skip_existing=True,
        exclude_urls=["wikipedia", "instagram.com", "aa.com", "facebook.com", "x.com"],
    )

    print("\n=== Результаты поиска ===")
    print(f"Обработано запросов: {stats['searched']}")
    print(f"Найдено ссылок: {stats['found']}")
    print(f"Новых добавлено: {stats['new']}")
    print(f"Пропущено (дубли): {stats['skipped']}")
    print(f"Исключено (фильтр): {stats['excluded']}")


if __name__ == "__main__":
    keywords_file = sys.argv[1] if len(sys.argv) > 1 else "keywords.txt"
    search_from_keywords_file(keywords_file)
