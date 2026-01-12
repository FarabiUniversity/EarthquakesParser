"""Example: Using the ContentParser."""

import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import pandas as pd
from dotenv import load_dotenv

from earthquakes_parser import SupabaseDB, SupabaseFileStorage
from earthquakes_parser.parser.parser_manager import ParserManager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    load_dotenv()
    db = SupabaseDB()
    file_storage = SupabaseFileStorage(bucket_name="html-files")
    parser_manager = ParserManager(db, file_storage)
    parser_manager.parse_downloaded()


if __name__ == "__main__":
    main()
