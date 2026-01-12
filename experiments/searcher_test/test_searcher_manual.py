"""Manual test script for KeywordSearcher using SearchManager and Supabase."""

import sys
from pathlib import Path

from dotenv import load_dotenv

from earthquakes_parser import SupabaseDB, SupabaseFileStorage
from earthquakes_parser.search import GoogleSearcher, SearchManager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Run manual tests for KeywordSearcher with Supabase integration."""
    print("\n" + "=" * 60)
    print("KEYWORD SEARCHER MANUAL TESTING (Supabase + GoogleSearcher)")
    print("=" * 60)

    # Initialize environment and components
    load_dotenv()
    searcher = GoogleSearcher(delay=1.0)
    database = SupabaseDB()
    search_manager = SearchManager(db=database, searcher=searcher)

    # Keywords to test
    keywords = [
        "землетрясение",
        "магнитуда землетрясение",
        "эпицентр землетрясение",
    ]

    # Perform search and save results to Supabase DB
    print("\nSearching and saving results...")
    search_results_stat = search_manager.search_and_save(
        keywords=keywords, max_results=2
    )
    print(search_results_stat)

    # Download HTML pages and save to Supabase storage
    print("\nDownloading HTML pages with Selenium...")
    file_storage = SupabaseFileStorage(bucket_name="html-files")
    download_results_stat = search_manager.download_html(
        file_storage, fetch_with="selenium"
    )
    print(download_results_stat)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
