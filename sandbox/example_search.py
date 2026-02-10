"""Test script: Search and save results to CSV file."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from earthquakes_parser.search import GoogleSearcher  # noqa: E402


def main():
    """Search for earthquake-related keywords and save to CSV."""
    # Initialize components
    load_dotenv()

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Initialize searcher
    searcher = GoogleSearcher(delay=1.0)

    # Keywords to search
    keywords = [
        "землетрясение в алматы",
        "магнитуда землетрясение в алматы",
        "эпицентр землетрясение в алматы",
    ]

    print(f"Searching for {len(keywords)} keywords...")

    # Perform search and convert to DataFrame
    df = searcher.search_to_dataframe(
        keywords=keywords, max_results=10  # 5 results per keyword
    )

    # Save to CSV
    output_path = data_dir / "links.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nSearch completed!")
    print(f"Total results found: {len(df)}")
    print(f"Results saved to: {output_path}")
    print("\nFirst few results:")
    print(df.head())


if __name__ == "__main__":
    main()
