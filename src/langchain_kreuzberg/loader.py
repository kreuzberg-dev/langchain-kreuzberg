"""Kreuzberg document loader for LangChain."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from kreuzberg import (
    ExtractionConfig,
    ExtractionResult,
    KreuzbergError,
    OcrConfig,
    PageConfig,
    extract_bytes,
    extract_bytes_sync,
    extract_file,
    extract_file_sync,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class KreuzbergLoader(BaseLoader):
    """Load documents using Kreuzberg, supporting 75+ file formats with true async.

    Kreuzberg is a Rust-powered document intelligence library. This loader wraps its
    extraction API to provide LangChain-compatible Documents with rich metadata.

    Examples:
        Load a single file:
            >>> loader = KreuzbergLoader(file_path="document.pdf")
            >>> docs = loader.load()

        Load from bytes:
            >>> loader = KreuzbergLoader(data=raw_bytes, mime_type="application/pdf")
            >>> docs = loader.load()

        Load a directory:
            >>> loader = KreuzbergLoader(file_path="./docs/", glob="**/*.pdf")
            >>> docs = loader.load()

        Per-page splitting:
            >>> loader = KreuzbergLoader(file_path="document.pdf", per_page=True)
            >>> docs = loader.load()  # One Document per page

        Async loading:
            >>> loader = KreuzbergLoader(file_path="document.pdf")
            >>> docs = await loader.aload()

    """

    def __init__(
        self,
        file_path: str | Path | list[str | Path] | None = None,
        *,
        data: bytes | None = None,
        mime_type: str | None = None,
        glob: str | None = None,
        output_format: str = "markdown",
        ocr_backend: str | None = None,
        ocr_language: str | None = None,
        force_ocr: bool = False,
        extract_tables: bool = True,
        per_page: bool = False,
        config: ExtractionConfig | None = None,
    ) -> None:
        """Initialize the KreuzbergLoader.

        Args:
            file_path: File path, list of file paths, or directory path to load.
            data: Raw bytes to extract text from. Mutually exclusive with file_path.
            mime_type: MIME type hint. Required when using data, optional for file_path.
            glob: Glob pattern for directory mode. Defaults to None (matches all files).
            output_format: Output format for extraction. One of "plain", "markdown",
                "djot", "html", "structured". Defaults to "markdown".
            ocr_backend: OCR backend name. One of "tesseract", "easyocr", "paddleocr".
            ocr_language: OCR language code (ISO 639-3, e.g., "eng", "deu", "fra").
            force_ocr: Force OCR even on searchable PDFs. Defaults to False.
            extract_tables: Include tables in page_content and metadata. Defaults to True.
            per_page: Yield one Document per page instead of one per file. Defaults to False.
            config: Full ExtractionConfig override. When provided, individual extraction
                parameters (output_format, ocr_backend, etc.) are ignored.

        Raises:
            ValueError: If neither file_path nor data is provided.
            ValueError: If both file_path and data are provided.
            ValueError: If data is provided without mime_type.

        """
        if file_path is None and data is None:
            msg = "Either 'file_path' or 'data' must be provided."
            raise ValueError(msg)
        if file_path is not None and data is not None:
            msg = "Cannot specify both 'file_path' and 'data'. Use one or the other."
            raise ValueError(msg)
        if data is not None and mime_type is None:
            msg = "'mime_type' is required when using 'data'."
            raise ValueError(msg)

        # Normalize file_path
        if isinstance(file_path, (str, Path)):
            self._file_path: Path | list[Path] | None = Path(file_path)
        elif file_path is not None:
            self._file_path = [Path(p) for p in file_path]
        else:
            self._file_path = None

        self._data = data
        self._mime_type = mime_type
        self._glob = glob
        self._output_format = output_format
        self._ocr_backend = ocr_backend
        self._ocr_language = ocr_language
        self._force_ocr = force_ocr
        self._extract_tables = extract_tables
        self._per_page = per_page
        self._config = config

    def _build_config(self) -> ExtractionConfig:
        """Build an ExtractionConfig from individual parameters.

        If a full config override was provided, return it directly.
        """
        if self._config is not None:
            return self._config

        kwargs: dict[str, Any] = {"output_format": self._output_format}

        if self._ocr_backend is not None or self._ocr_language is not None:
            ocr_kwargs: dict[str, Any] = {}
            if self._ocr_backend is not None:
                ocr_kwargs["backend"] = self._ocr_backend
            if self._ocr_language is not None:
                ocr_kwargs["language"] = self._ocr_language
            kwargs["ocr"] = OcrConfig(**ocr_kwargs)

        if self._force_ocr:
            kwargs["force_ocr"] = True

        if self._per_page:
            kwargs["pages"] = PageConfig(extract_pages=True)

        return ExtractionConfig(**kwargs)

    def _result_to_documents(self, result: ExtractionResult, source: str) -> Iterator[Document]:
        """Convert an ExtractionResult to one or more LangChain Documents."""
        if self._per_page and result.pages:
            yield from self._pages_to_documents(result, source)
        else:
            metadata = self._build_metadata(result, source)
            page_content = self._assemble_content(result.content, result.tables)
            yield Document(page_content=page_content, metadata=metadata)

    def _build_metadata(self, result: ExtractionResult, source: str) -> dict[str, Any]:
        """Build a flat metadata dict from an ExtractionResult."""
        metadata: dict[str, Any] = {}

        # Flatten Kreuzberg metadata (a TypedDict / plain dict)
        if isinstance(result.metadata, dict):
            metadata.update({k: v for k, v in result.metadata.items() if v is not None})

        # Top-level enrichment
        metadata["mime_type"] = result.mime_type
        if result.quality_score is not None:
            metadata["quality_score"] = result.quality_score
        if result.detected_languages is not None:
            metadata["detected_languages"] = result.detected_languages
        if result.output_format is not None:
            metadata["output_format"] = result.output_format
        metadata["page_count"] = result.get_page_count()

        # Extracted keywords
        if result.extracted_keywords:
            metadata["extracted_keywords"] = [
                {
                    "text": kw.text,
                    "score": kw.score,
                    "algorithm": kw.algorithm,
                }
                for kw in result.extracted_keywords
            ]

        # Tables metadata
        metadata["table_count"] = len(result.tables)
        if self._extract_tables and result.tables:
            metadata["tables"] = [
                {
                    "cells": table.cells,
                    "markdown": table.markdown,
                    "page_number": table.page_number,
                }
                for table in result.tables
            ]

        # Processing warnings
        if result.processing_warnings:
            metadata["processing_warnings"] = [str(w) for w in result.processing_warnings]

        metadata["source"] = source

        return metadata

    def _pages_to_documents(
        self,
        result: ExtractionResult,
        source: str,
    ) -> Iterator[Document]:
        """Yield one Document per page from an ExtractionResult."""
        base_metadata = self._build_metadata(result, source)

        for page in result.pages:
            page_metadata = {**base_metadata}

            # Page-specific fields (Kreuzberg uses 1-indexed, LangChain uses 0-indexed)
            page_number: int = page["page_number"]
            page_metadata["page"] = page_number - 1
            if page.get("is_blank") is not None:
                page_metadata["is_blank"] = page["is_blank"]

            # Assemble page content
            page_tables = page.get("tables", [])
            page_content = self._assemble_content(page["content"], page_tables)

            yield Document(page_content=page_content, metadata=page_metadata)

    def _assemble_content(
        self,
        content: str,
        tables: Any,
    ) -> str:
        """Combine text content with table markdown if extract_tables is enabled."""
        if not self._extract_tables or not tables:
            return content

        table_parts = [table.markdown if hasattr(table, "markdown") else table.get("markdown", "") for table in tables]
        return "\n\n".join([content, *(m for m in table_parts if m)])

    def _resolve_file_paths(self) -> Iterator[Path]:
        """Resolve file paths from the configured file_path."""
        if isinstance(self._file_path, list):
            yield from self._file_path
        elif isinstance(self._file_path, Path):
            if self._file_path.is_dir():
                pattern = self._glob or "**/*"
                yield from (p for p in self._file_path.glob(pattern) if p.is_file())
            else:
                yield self._file_path

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily, yielding one Document at a time.

        Yields:
            Document objects with extracted text and metadata.

        """
        config = self._build_config()

        if self._data is not None:
            mime_type: str = self._mime_type  # type: ignore[assignment]  # Validated in __init__
            result = extract_bytes_sync(self._data, mime_type, config=config)
            source = f"bytes://{mime_type}"
            yield from self._result_to_documents(result, source)
        else:
            for path in self._resolve_file_paths():
                try:
                    result = extract_file_sync(path, mime_type=self._mime_type, config=config)
                except KreuzbergError as exc:
                    msg = f"Failed to extract '{path}': {exc}"
                    raise type(exc)(msg) from exc
                yield from self._result_to_documents(result, str(path))

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load documents asynchronously, yielding one Document at a time.

        Uses Kreuzberg's native async extraction backed by Rust's tokio runtime.

        Yields:
            Document objects with extracted text and metadata.

        """
        config = self._build_config()

        if self._data is not None:
            mime_type: str = self._mime_type  # type: ignore[assignment]  # Validated in __init__
            result = await extract_bytes(self._data, mime_type, config=config)
            source = f"bytes://{mime_type}"
            for doc in self._result_to_documents(result, source):
                yield doc
        else:
            for path in self._resolve_file_paths():
                try:
                    result = await extract_file(path, mime_type=self._mime_type, config=config)
                except KreuzbergError as exc:
                    msg = f"Failed to extract '{path}': {exc}"
                    raise type(exc)(msg) from exc
                for doc in self._result_to_documents(result, str(path)):
                    yield doc
