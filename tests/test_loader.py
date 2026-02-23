"""Synchronous test suite for KreuzbergLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from kreuzberg import ExtractionConfig, OcrConfig

from langchain_kreuzberg import KreuzbergLoader
from tests.conftest import make_mock_keyword, make_mock_result, make_mock_table


class TestConstructorValidation:
    """Test constructor input validation."""

    def test_no_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Either 'file_path' or 'data'"):
            KreuzbergLoader()

    def test_both_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            KreuzbergLoader(file_path="test.pdf", data=b"test")

    def test_bytes_requires_mime_type(self) -> None:
        with pytest.raises(ValueError, match="'mime_type' is required"):
            KreuzbergLoader(data=b"test")

    def test_valid_file_path(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf")
        assert loader._file_path == Path("test.pdf")

    def test_valid_bytes_input(self) -> None:
        loader = KreuzbergLoader(data=b"test", mime_type="text/plain")
        assert loader._data == b"test"
        assert loader._mime_type == "text/plain"

    def test_valid_multiple_files(self) -> None:
        loader = KreuzbergLoader(file_path=["a.pdf", "b.docx"])
        assert loader._file_path == [Path("a.pdf"), Path("b.docx")]

    def test_valid_path_object(self) -> None:
        loader = KreuzbergLoader(file_path=Path("test.pdf"))
        assert loader._file_path == Path("test.pdf")


class TestBuildConfig:
    """Test ExtractionConfig construction from parameters."""

    def test_default_config(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf")
        config = loader._build_config()
        assert isinstance(config, ExtractionConfig)
        assert config.output_format == "markdown"

    def test_output_format_passthrough(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", output_format="plain")
        config = loader._build_config()
        assert config.output_format == "plain"

    def test_ocr_config_construction(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", ocr_backend="tesseract", ocr_language="deu")
        config = loader._build_config()
        assert config.ocr is not None
        assert config.ocr.backend == "tesseract"
        assert config.ocr.language == "deu"

    def test_ocr_backend_only(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", ocr_backend="easyocr")
        config = loader._build_config()
        assert config.ocr is not None
        assert config.ocr.backend == "easyocr"

    def test_ocr_language_only(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", ocr_language="fra")
        config = loader._build_config()
        assert config.ocr is not None
        assert config.ocr.language == "fra"

    def test_force_ocr_passthrough(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", force_ocr=True)
        config = loader._build_config()
        assert config.force_ocr is True

    def test_per_page_creates_page_config(self) -> None:
        loader = KreuzbergLoader(file_path="test.pdf", per_page=True)
        config = loader._build_config()
        assert config.pages is not None
        assert config.pages.extract_pages is True

    def test_config_override(self) -> None:
        custom_config = ExtractionConfig(
            output_format="html",
            force_ocr=True,
            ocr=OcrConfig(backend="paddleocr"),
        )
        loader = KreuzbergLoader(
            file_path="test.pdf",
            output_format="plain",  # Should be ignored
            config=custom_config,
        )
        config = loader._build_config()
        assert config is custom_config
        assert config.output_format == "html"


class TestSyncLoading:
    """Test synchronous loading via lazy_load()."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_single_text_file(self, mock_extract: MagicMock, sample_txt_path: Path) -> None:
        mock_extract.return_value = make_mock_result(
            content="Sample text content",
            metadata={"format_type": "text", "word_count": 5},
        )

        loader = KreuzbergLoader(file_path=str(sample_txt_path))
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Sample text content"
        assert docs[0].metadata["source"] == str(sample_txt_path)
        mock_extract.assert_called_once()

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_single_pdf(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            content="PDF content",
            mime_type="application/pdf",
            metadata={
                "format_type": "pdf",
                "pdf_version": "1.7",
                "producer": "Test",
                "page_count": 3,
            },
            page_count=3,
        )

        loader = KreuzbergLoader(file_path="document.pdf")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "PDF content"
        assert docs[0].metadata["mime_type"] == "application/pdf"
        assert docs[0].metadata["format_type"] == "pdf"
        assert docs[0].metadata["pdf_version"] == "1.7"
        assert docs[0].metadata["page_count"] == 3

    @patch("langchain_kreuzberg.loader.extract_bytes_sync")
    def test_load_bytes_mode(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(content="Bytes content")

        loader = KreuzbergLoader(data=b"raw data", mime_type="text/plain")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Bytes content"
        assert docs[0].metadata["source"] == "bytes://text/plain"
        mock_extract.assert_called_once()

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_multiple_files(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path=["a.txt", "b.txt", "c.txt"])
        docs = loader.load()

        assert len(docs) == 3
        assert mock_extract.call_count == 3
        sources = [d.metadata["source"] for d in docs]
        assert sources == ["a.txt", "b.txt", "c.txt"]

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_directory_with_glob(self, mock_extract: MagicMock, tmp_dir_with_files: Path) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path=str(tmp_dir_with_files), glob="*.txt")
        docs = loader.load()

        # Only top-level .txt files (file1.txt, file2.txt)
        assert len(docs) == 2
        assert mock_extract.call_count == 2

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_directory_default_glob(self, mock_extract: MagicMock, tmp_dir_with_files: Path) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path=str(tmp_dir_with_files))
        docs = loader.load()

        # Default glob **/* matches all files including subdir/file3.txt
        assert len(docs) == 3
        assert mock_extract.call_count == 3

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_load_empty_directory(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        loader = KreuzbergLoader(file_path=str(tmp_path))
        docs = loader.load()

        assert len(docs) == 0
        mock_extract.assert_not_called()


class TestPerPageSplitting:
    """Test per-page document splitting."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_per_page_splitting(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            pages=[
                {"page_number": 1, "content": "Page 1 text", "tables": [], "images": [], "is_blank": False},
                {"page_number": 2, "content": "Page 2 text", "tables": [], "images": [], "is_blank": False},
                {"page_number": 3, "content": "", "tables": [], "images": [], "is_blank": True},
            ],
            page_count=3,
        )

        loader = KreuzbergLoader(file_path="doc.pdf", per_page=True)
        docs = loader.load()

        assert len(docs) == 3
        assert docs[0].page_content == "Page 1 text"
        assert docs[1].page_content == "Page 2 text"
        assert docs[2].page_content == ""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_per_page_metadata(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            pages=[
                {"page_number": 1, "content": "Page 1", "tables": [], "images": [], "is_blank": False},
                {"page_number": 2, "content": "Page 2", "tables": [], "images": [], "is_blank": True},
            ],
            page_count=2,
        )

        loader = KreuzbergLoader(file_path="doc.pdf", per_page=True)
        docs = loader.load()

        # Page numbers are 0-indexed in LangChain convention
        assert docs[0].metadata["page"] == 0
        assert docs[0].metadata["is_blank"] is False
        assert docs[1].metadata["page"] == 1
        assert docs[1].metadata["is_blank"] is True

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_per_page_with_tables(self, mock_extract: MagicMock) -> None:
        page_table = {"markdown": "| X |\n|---|\n| Y |"}
        mock_extract.return_value = make_mock_result(
            pages=[
                {
                    "page_number": 1,
                    "content": "Text",
                    "tables": [page_table],
                    "images": [],
                    "is_blank": False,
                },
            ],
            page_count=1,
        )

        loader = KreuzbergLoader(file_path="doc.pdf", per_page=True)
        docs = loader.load()

        assert "| X |" in docs[0].page_content

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_per_page_fallback_when_no_pages(self, mock_extract: MagicMock) -> None:
        """When per_page=True but result has no pages, fall back to whole document."""
        mock_extract.return_value = make_mock_result(content="Whole document", pages=None)

        loader = KreuzbergLoader(file_path="doc.txt", per_page=True)
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Whole document"


class TestMetadata:
    """Test metadata extraction and flattening."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_metadata_source_key(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path="doc.txt")
        docs = loader.load()

        assert "source" in docs[0].metadata
        assert docs[0].metadata["source"] == "doc.txt"

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_metadata_flattening(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            metadata={
                "format_type": "text",
                "title": "Test Doc",
                "authors": ["Alice", "Bob"],
                "keywords": None,  # Should be dropped
            },
        )

        loader = KreuzbergLoader(file_path="doc.txt")
        docs = loader.load()

        meta = docs[0].metadata
        assert meta["format_type"] == "text"
        assert meta["title"] == "Test Doc"
        assert meta["authors"] == ["Alice", "Bob"]
        assert "keywords" not in meta  # None values dropped

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_metadata_enrichment(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            quality_score=0.85,
            detected_languages=["eng", "deu"],
            output_format="markdown",
        )

        loader = KreuzbergLoader(file_path="doc.txt")
        docs = loader.load()

        meta = docs[0].metadata
        assert meta["quality_score"] == 0.85
        assert meta["detected_languages"] == ["eng", "deu"]
        assert meta["output_format"] == "markdown"
        assert meta["mime_type"] == "text/plain"
        assert meta["page_count"] == 1

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_extracted_keywords_in_metadata(self, mock_extract: MagicMock) -> None:
        kw1 = make_mock_keyword(text="python", score=0.95, algorithm="yake")
        kw2 = make_mock_keyword(text="machine learning", score=0.88, algorithm="yake")
        mock_extract.return_value = make_mock_result(extracted_keywords=[kw1, kw2])

        loader = KreuzbergLoader(file_path="doc.txt")
        docs = loader.load()

        keywords = docs[0].metadata["extracted_keywords"]
        assert len(keywords) == 2
        assert keywords[0] == {"text": "python", "score": 0.95, "algorithm": "yake"}
        assert keywords[1]["text"] == "machine learning"

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_processing_warnings_in_metadata(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result(
            processing_warnings=["Low quality scan detected", "Missing font fallback"]
        )

        loader = KreuzbergLoader(file_path="doc.txt")
        docs = loader.load()

        assert "processing_warnings" in docs[0].metadata
        assert len(docs[0].metadata["processing_warnings"]) == 2


class TestTableExtraction:
    """Test table handling in content and metadata."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_table_extraction_in_content(self, mock_extract: MagicMock) -> None:
        table = make_mock_table(markdown="| Col1 | Col2 |\n|---|---|\n| A | B |")
        mock_extract.return_value = make_mock_result(content="Main text", tables=[table])

        loader = KreuzbergLoader(file_path="doc.pdf")
        docs = loader.load()

        assert "Main text" in docs[0].page_content
        assert "| Col1 | Col2 |" in docs[0].page_content

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_table_extraction_in_metadata(self, mock_extract: MagicMock) -> None:
        table = make_mock_table(
            cells=[["A", "B"], ["1", "2"]],
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            page_number=1,
        )
        mock_extract.return_value = make_mock_result(tables=[table])

        loader = KreuzbergLoader(file_path="doc.pdf")
        docs = loader.load()

        meta = docs[0].metadata
        assert meta["table_count"] == 1
        assert len(meta["tables"]) == 1
        assert meta["tables"][0]["cells"] == [["A", "B"], ["1", "2"]]
        assert meta["tables"][0]["page_number"] == 1

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_tables_disabled(self, mock_extract: MagicMock) -> None:
        table = make_mock_table()
        mock_extract.return_value = make_mock_result(content="Main text", tables=[table])

        loader = KreuzbergLoader(file_path="doc.pdf", extract_tables=False)
        docs = loader.load()

        # Table markdown should NOT be in page_content
        assert docs[0].page_content == "Main text"
        # Table structured data should NOT be in metadata
        assert "tables" not in docs[0].metadata
        # But table_count should still be present
        assert docs[0].metadata["table_count"] == 1

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_multiple_tables_in_content(self, mock_extract: MagicMock) -> None:
        t1 = make_mock_table(markdown="| T1 |")
        t2 = make_mock_table(markdown="| T2 |")
        mock_extract.return_value = make_mock_result(content="Text", tables=[t1, t2])

        loader = KreuzbergLoader(file_path="doc.pdf")
        docs = loader.load()

        # Tables separated by double newlines
        assert docs[0].page_content == "Text\n\n| T1 |\n\n| T2 |"


class TestErrorPropagation:
    """Test that Kreuzberg errors propagate correctly."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_error_propagation(self, mock_extract: MagicMock) -> None:
        from kreuzberg.exceptions import KreuzbergError

        mock_extract.side_effect = KreuzbergError("Extraction failed")

        loader = KreuzbergLoader(file_path="bad.pdf")

        with pytest.raises(KreuzbergError, match=r"Failed to extract 'bad\.pdf'"):
            loader.load()


class TestLazyLoad:
    """Test that lazy_load returns an iterator."""

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_lazy_load_is_iterator(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path="doc.txt")
        result = loader.lazy_load()

        # Should be an iterator, not a list
        assert hasattr(result, "__next__")

    @patch("langchain_kreuzberg.loader.extract_file_sync")
    def test_lazy_load_yields_documents(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = make_mock_result()

        loader = KreuzbergLoader(file_path=["a.txt", "b.txt"])

        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        assert len(docs) == 2
