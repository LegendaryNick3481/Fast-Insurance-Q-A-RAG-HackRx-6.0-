"""
utils.py
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from voyageai import Client
from typing import List
from pathlib import Path
import httpx
import os
import tempfile
from starlette.concurrency import run_in_threadpool

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment")

voyage = Client(api_key=VOYAGE_API_KEY)

def get_splitter(num_pages: int) -> RecursiveCharacterTextSplitter:
    if num_pages > 500:
        chunk_size, chunk_overlap = 1400, 350
    elif num_pages > 300:
        chunk_size, chunk_overlap = 1200, 300
    elif num_pages > 200:
        chunk_size, chunk_overlap = 1000, 250
    elif num_pages > 100:
        chunk_size, chunk_overlap = 900, 200
    elif num_pages > 50:
        chunk_size, chunk_overlap = 800, 180
    elif num_pages > 20:
        chunk_size, chunk_overlap = 700, 150
    elif num_pages > 10:
        chunk_size, chunk_overlap = 600, 120
    else:
        chunk_size, chunk_overlap = 500, 100

    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

async def load_pdf(path_or_url: str) -> List[Document]:
    temp_download = False

    if path_or_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient() as client:
            response = await client.get(path_or_url)
            response.raise_for_status()
            file_path = Path(tempfile.mkstemp(suffix=".pdf")[1])
            with open(file_path, "wb") as f:
                f.write(response.content)
            temp_download = True
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def blocking_load():
        loader = PyPDFLoader(str(file_path))
        raw_docs = loader.load()
        splitter = get_splitter(num_pages=len(raw_docs))
        return splitter.split_documents(raw_docs)

    split_docs = await run_in_threadpool(blocking_load)

    if temp_download:
        os.remove(file_path)

    return split_docs

async def rerank_documents(query: str, retrieved_docs: List[Document], top_n: int = 5) -> List[Document]:
    """
    Depreciated Function. We now use VoyageAIRerankRetriever (Integrated with langchain)
    """
    docs_texts = [doc.page_content for doc in retrieved_docs]

    def blocking_rerank():
        response = voyage.rerank(
            query=query,
            documents=docs_texts,
            model="rerank-2.5"
        )
        return sorted(response.results, key=lambda x: x.relevance_score, reverse=True)[:top_n]

    top_results = await run_in_threadpool(blocking_rerank)

    return [retrieved_docs[result.index] for result in top_results]
