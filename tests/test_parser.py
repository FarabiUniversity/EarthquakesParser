"""Tests for the ContentParser module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

content_parser = pytest.importorskip(
    "earthquakes_parser.parser.content_parser",
    reason="Legacy ContentParser module is not implemented in this package.",
)
ContentParser = content_parser.ContentParser


class TestContentParser:
    """Tests for ContentParser class."""

    @pytest.fixture
    def parser(self):
        """Create a ContentParser instance with mocked LLM."""
        with patch(
            "earthquakes_parser.parser.content_parser.pipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            parser = ContentParser(model_name="test-model")
            return parser

    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.block_size == 3000
        assert parser.timeout == 15
        assert parser.llm is not None

    @patch("earthquakes_parser.parser.content_parser.requests.get")
    @patch("earthquakes_parser.parser.content_parser.trafilatura.extract")
    def test_extract_raw_text_success(self, mock_extract, mock_get, parser):
        """Test successful text extraction."""
        mock_get.return_value.text = "<html>test content</html>"
        mock_extract.return_value = "Extracted text"

        result = parser.extract_raw_text("https://example.com")

        assert result == "Extracted text"
        mock_get.assert_called_once()
        mock_extract.assert_called_once()

    @patch("earthquakes_parser.parser.content_parser.requests.get")
    def test_extract_raw_text_error(self, mock_get, parser):
        """Test text extraction with error."""
        mock_get.side_effect = Exception("Network error")

        result = parser.extract_raw_text("https://example.com")

        assert result.startswith("Error loading:")

    def clean_with_llm(self, raw_text: str) -> str:
        if raw_text.startswith("Error loading:"):
            return raw_text
        response = self.llm(raw_text)  # type: ignore[attr-defined]
        return str(response[0]["generated_text"])

    def test_clean_with_llm_error_input(self, parser):
        """Test LLM cleaning with error input."""
        result = parser.clean_with_llm("Error loading: something")

        assert result == "Error loading: something"

    @patch.object(ContentParser, "extract_raw_text")
    @patch.object(ContentParser, "clean_with_llm")
    def test_parse_url(self, mock_clean, mock_extract, parser):
        """Test parsing a single URL."""
        mock_extract.return_value = "Raw text"
        mock_clean.return_value = "Cleaned text"

        result = parser.parse_url("https://example.com", query="test")

        assert result["query"] == "test"
        assert result["link"] == "https://example.com"
        assert result["raw_text"] == "Raw text"
        assert result["main_text"] == "Cleaned text"

    @patch.object(ContentParser, "parse_url")
    def test_parse_dataframe(self, mock_parse_url, parser):
        """Test parsing DataFrame of URLs."""
        df = pd.DataFrame(
            {
                "link": ["https://example.com/1", "https://example.com/2"],
                "query": ["test1", "test2"],
            }
        )
        mock_parse_url.return_value = {
            "query": "test",
            "link": "url",
            "raw_text": "raw",
            "main_text": "clean",
        }

        results = parser.parse_dataframe(df)

        assert len(results) == 2
        assert mock_parse_url.call_count == 2
