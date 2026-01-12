"""Parser module for extracting content from web pages."""

from earthquakes_parser.parser.data_extractor import DataExtractor
from earthquakes_parser.parser.models import ExtractionResult, PageSchema, ParsedContent
from earthquakes_parser.parser.parser_manager import ParserManager
from earthquakes_parser.parser.schema_extractor import SchemaExtractor
from earthquakes_parser.parser.schema_manager import SchemaManager

__all__ = [
    "ParserManager",
    "SchemaExtractor",
    "SchemaManager",
    "DataExtractor",
    "PageSchema",
    "ParsedContent",
    "ExtractionResult",
]