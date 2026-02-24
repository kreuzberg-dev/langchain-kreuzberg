# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-23

### Added

- `KreuzbergLoader` class extending `langchain_core.document_loaders.BaseLoader`
- Support for 75+ file formats via Kreuzberg extraction API
- Synchronous loading via `load()` and `lazy_load()` methods
- Native async loading via `aload()` and `alazy_load()` backed by Rust's tokio runtime
- File path input supporting single files, lists of files, and directories with glob patterns
- Raw bytes input for loading from API responses, S3 objects, and other in-memory sources
- Rich metadata extraction including title, author, page count, quality score, detected languages, and extracted keywords
- Table extraction with cell data and Markdown representation
- Per-page splitting mode yielding one `Document` per page for RAG pipelines
- OCR support with three configurable backends: Tesseract, EasyOCR, and PaddleOCR
- OCR language configuration via ISO 639-3 codes
- Force OCR option for searchable PDFs
- Output format selection: plain text, Markdown, Djot, HTML, and structured
- Full `ExtractionConfig` override for advanced configuration
- Processing warnings surfaced in document metadata
- Type annotations and `py.typed` marker for static analysis
- CI pipeline with linting (Ruff), type checking (mypy), and testing (pytest) across Python 3.10, 3.12, and 3.13

[0.1.0]: https://github.com/Goldziher/langchain-kreuzberg/releases/tag/v0.1.0
