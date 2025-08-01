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
    raise ValueError("❌ VOYAGE_API_KEY not found in environment")

voyage = Client(api_key=VOYAGE_API_KEY)

def get_splitter(num_pages: int) -> RecursiveCharacterTextSplitter:
    if num_pages > 300:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    elif num_pages > 100:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

def load_pdf(path_or_url: str) -> List[Document]:
    print(f"🔧 Starting load_pdf() for: {path_or_url}", flush=True)

    # Step 1: Download if URL
    if path_or_url.startswith(("http://", "https://")):
        try:
            print("🌐 Detected URL. Downloading PDF...", flush=True)
            response = requests.get(path_or_url)
            response.raise_for_status()
            file_path = Path("temp.pdf")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("✅ PDF downloaded to temp.pdf.", flush=True)
        except Exception as e:
            print(f"❌ Error downloading PDF from URL: {e}", flush=True)
            raise
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}", flush=True)
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"📄 Using local PDF file: {file_path}", flush=True)

    # Step 2: Load with PyPDFLoader
    try:
        print("📥 Loading PDF with PyPDFLoader...", flush=True)
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} raw pages from PDF.", flush=True)
    except Exception as e:
        print(f"❌ Error loading PDF: {e}", flush=True)
        raise

    # Step 3: Split using dynamic splitter
    try:
        print("🔍 Splitting documents into chunks...", flush=True)
        splitter = get_splitter(num_pages=len(docs))
        split_docs = splitter.split_documents(docs)
        print(f"✅ Split into {len(split_docs)} chunks.", flush=True)
    except Exception as e:
        print(f"❌ Error splitting documents: {e}", flush=True)
        raise

    return split_docs
