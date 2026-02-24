"""Asynchronous test suite for KreuzbergLoader."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from kreuzberg import ExtractionConfig, PageConfig

from langchain_kreuzberg import KreuzbergLoader
from tests.conftest import make_mock_result


@patch("langchain_kreuzberg.loader.extract_file", new_callable=AsyncMock)
async def test_alazy_load_single_file(mock_extract: AsyncMock) -> None:
    mock_extract.return_value = make_mock_result(content="Async content")

    loader = KreuzbergLoader(file_path="doc.txt")
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1
    assert docs[0].page_content == "Async content"
    assert docs[0].metadata["source"] == "doc.txt"
    mock_extract.assert_called_once()


@patch("langchain_kreuzberg.loader.extract_bytes", new_callable=AsyncMock)
async def test_alazy_load_bytes_mode(mock_extract: AsyncMock) -> None:
    mock_extract.return_value = make_mock_result(content="Bytes async content")

    loader = KreuzbergLoader(data=b"raw data", mime_type="text/plain")
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1
    assert docs[0].page_content == "Bytes async content"
    assert docs[0].metadata["source"] == "bytes://text/plain"
    mock_extract.assert_called_once()


@patch("langchain_kreuzberg.loader.batch_extract_files", new_callable=AsyncMock)
async def test_alazy_load_multiple_files(mock_batch: AsyncMock) -> None:
    mock_batch.return_value = [make_mock_result(), make_mock_result(), make_mock_result()]

    loader = KreuzbergLoader(file_path=["a.txt", "b.txt", "c.txt"])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 3
    mock_batch.assert_called_once()


@patch("langchain_kreuzberg.loader.batch_extract_files", new_callable=AsyncMock)
async def test_alazy_load_directory(mock_batch: AsyncMock, tmp_dir_with_files: Path) -> None:
    mock_batch.return_value = [make_mock_result(), make_mock_result()]

    loader = KreuzbergLoader(file_path=str(tmp_dir_with_files), glob="*.txt")
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    # Only top-level .txt files
    assert len(docs) == 2


async def test_alazy_load_empty_directory(tmp_path: Path) -> None:
    loader = KreuzbergLoader(file_path=str(tmp_path))
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 0


@patch("langchain_kreuzberg.loader.extract_file", new_callable=AsyncMock)
async def test_alazy_load_per_page(mock_extract: AsyncMock) -> None:
    mock_extract.return_value = make_mock_result(
        pages=[
            {"page_number": 1, "content": "Page 1", "tables": [], "images": [], "is_blank": False},
            {"page_number": 2, "content": "Page 2", "tables": [], "images": [], "is_blank": False},
        ],
        page_count=2,
    )

    config = ExtractionConfig(pages=PageConfig(extract_pages=True))
    loader = KreuzbergLoader(file_path="doc.pdf", config=config)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 2
    assert docs[0].page_content == "Page 1"
    assert docs[0].metadata["page"] == 0
    assert docs[1].page_content == "Page 2"
    assert docs[1].metadata["page"] == 1


@patch("langchain_kreuzberg.loader.extract_file", new_callable=AsyncMock)
async def test_aload_convenience(mock_extract: AsyncMock) -> None:
    mock_extract.return_value = make_mock_result(content="Async doc")

    loader = KreuzbergLoader(file_path="doc.txt")
    docs = await loader.aload()

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].page_content == "Async doc"


@patch("langchain_kreuzberg.loader.extract_file", new_callable=AsyncMock)
async def test_alazy_load_error_propagation(mock_extract: AsyncMock) -> None:
    from kreuzberg.exceptions import KreuzbergError

    mock_extract.side_effect = KreuzbergError("Async extraction failed")

    loader = KreuzbergLoader(file_path="bad.pdf")

    with pytest.raises(KreuzbergError, match=r"Failed to extract 'bad\.pdf'"):
        async for _doc in loader.alazy_load():
            pass


@patch("langchain_kreuzberg.loader.batch_extract_files", new_callable=AsyncMock)
async def test_alazy_load_batch_error_propagation(mock_batch: AsyncMock) -> None:
    from kreuzberg.exceptions import KreuzbergError

    error_result = make_mock_result(content="Error: unsupported format", mime_type="text/plain", metadata={})
    mock_batch.return_value = [make_mock_result(), error_result]

    loader = KreuzbergLoader(file_path=["good.txt", "bad.xyz"])

    with pytest.raises(KreuzbergError, match=r"Failed to extract 'bad\.xyz'"):
        async for _doc in loader.alazy_load():
            pass
