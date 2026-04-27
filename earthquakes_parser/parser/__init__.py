"""Parser module for extracting content from web pages.

Keep this module lightweight.

Many submodules depend on optional third-party packages (e.g. `openai`).
Importing them at module import-time can break test collection/usage in
environments where those optional dependencies are not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from earthquakes_parser.parser.content_parser import ContentParser
    from earthquakes_parser.parser.data_extractor import DataExtractor
    from earthquakes_parser.parser.models import (
        ExtractionResult,
        PageSchema,
        ParsedContent,
    )
    from earthquakes_parser.parser.parser_manager import ParserManager
    from earthquakes_parser.parser.schema_extractor import SchemaExtractor
    from earthquakes_parser.parser.schema_manager import SchemaManager
    from earthquakes_parser.parser.tengrinews_parser import NewsArticle, TengriNewsParser

__all__ = [
    "ContentParser",
    "ParserManager",
    "SchemaExtractor",
    "SchemaManager",
    "DataExtractor",
    "PageSchema",
    "ParsedContent",
    "ExtractionResult",
    "TengriNewsParser",
    "NewsArticle",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "ContentParser": ("earthquakes_parser.parser.content_parser", "ContentParser"),
    "ParserManager": ("earthquakes_parser.parser.parser_manager", "ParserManager"),
    "SchemaExtractor": (
        "earthquakes_parser.parser.schema_extractor",
        "SchemaExtractor",
    ),
    "SchemaManager": ("earthquakes_parser.parser.schema_manager", "SchemaManager"),
    "DataExtractor": ("earthquakes_parser.parser.data_extractor", "DataExtractor"),
    "PageSchema": ("earthquakes_parser.parser.models", "PageSchema"),
    "ParsedContent": ("earthquakes_parser.parser.models", "ParsedContent"),
    "ExtractionResult": ("earthquakes_parser.parser.models", "ExtractionResult"),
    "TengriNewsParser": (
        "earthquakes_parser.parser.tengrinews_parser",
        "TengriNewsParser",
    ),
    "NewsArticle": ("earthquakes_parser.parser.tengrinews_parser", "NewsArticle"),
}


def __getattr__(name: str) -> Any:
    """Lazily import public symbols.

    This preserves the `from earthquakes_parser.parser import X` API while avoiding
    importing optional dependencies unless the symbol is actually requested.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
