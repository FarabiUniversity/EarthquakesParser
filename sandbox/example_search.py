"""Example: Using the KeywordSearcher."""

import sys
from pathlib import Path

from dotenv import load_dotenv

from earthquakes_parser import SupabaseDB, SupabaseFileStorage
from earthquakes_parser.search import GoogleSearcher, SearchManager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Demonstrate searching for earthquake-related keywords."""
    # Initialize components
    load_dotenv()
    searcher = GoogleSearcher(delay=1.0)
    keywords = [
        "землетрясение в алматы",
        "магнитуда землетрясение в алматы",
        "эпицентр землетрясение в алматы",
    ]

    database = SupabaseDB()
    search_manager = SearchManager(db=database, searcher=searcher)
    search_results_stat = search_manager.search_and_save(
        keywords=keywords, max_results=1
    )
    print(search_results_stat)

    file_storage = SupabaseFileStorage(bucket_name="html-files")
    download_results_stat = search_manager.download_html(
        file_storage, fetch_with="selenium"
    )
    print(download_results_stat)


if __name__ == "__main__":
    main()
