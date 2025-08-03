"""
Ultra-optimized utils.py - Maximum speed optimizations:
- Largest possible chunks to minimize embeddings
- Streaming PDF processing
- Aggressive filtering
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from pathlib import Path
import httpx
import os
import tempfile
import asyncio
from starlette.concurrency import run_in_threadpool

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment")

def get_ultra_fast_splitter(num_pages: int) -> RecursiveCharacterTextSplitter:
    if num_pages > 300:
        chunk_size, chunk_overlap = 4000, 600
    elif num_pages > 150:
        chunk_size, chunk_overlap = 3500, 500
    elif num_pages > 75:
        chunk_size, chunk_overlap = 3000, 400
    elif num_pages > 40:
        chunk_size, chunk_overlap = 2500, 350
    elif num_pages > 20:
        chunk_size, chunk_overlap = 2000, 300
    elif num_pages > 10:
        chunk_size, chunk_overlap = 1500, 250
    else:
        chunk_size, chunk_overlap = 1200, 200

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        length_function=len,
        is_separator_regex=False,
    )

async def load_pdf_ultra_fast(path_or_url: str) -> List[Document]:
    temp_download = False

    if path_or_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)
        ) as client:
            response = await client.get(path_or_url, follow_redirects=True)
            response.raise_for_status()

            file_path = Path(tempfile.mkstemp(suffix=".pdf")[1])
            with open(file_path, "wb") as f:
                f.write(response.content)
            temp_download = True
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def ultra_fast_load():
        try:
            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()
            if not raw_docs:
                return []

            filtered_docs = [
                doc for doc in raw_docs
                if len(doc.page_content.strip()) > 200 and doc.page_content.strip().count(' ') > 20
            ]

            if not filtered_docs:
                return []

            splitter = get_ultra_fast_splitter(len(filtered_docs))
            split_docs = splitter.split_documents(filtered_docs)

            final_docs = [
                doc for doc in split_docs
                if len(doc.page_content.strip()) > 300 and doc.page_content.strip().count(' ') > 30
            ]

            if len(final_docs) > 50:
                step = len(final_docs) / 50
                final_docs = [final_docs[int(i * step)] for i in range(50)]

            return final_docs

        except Exception:
            return []

    split_docs = await run_in_threadpool(ultra_fast_load)

    if temp_download and file_path.exists():
        try:
            os.remove(file_path)
        except:
            pass

    return split_docs

def cleanup_temp_files(*file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
