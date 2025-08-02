from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from voyageai import Client
from typing import List
import requests
from pathlib import Path
import os

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

def load_pdf(path_or_url: str) -> List[Document]:
    # Step 1: Download if URL
    if path_or_url.startswith(("http://", "https://")):
        try:
            response = requests.get(path_or_url)
            response.raise_for_status()
            file_path = Path("temp.pdf")
            with open(file_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Step 2: Load with PyPDFLoader
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    except Exception as e:
        raise

    # Step 3: Split using dynamic splitter
    try:
        splitter = get_splitter(num_pages=len(docs))
        split_docs = splitter.split_documents(docs)
    except Exception as e:
        raise

    return split_docs
