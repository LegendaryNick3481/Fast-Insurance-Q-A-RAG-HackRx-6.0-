"""
Enhanced RAG utils for insurance documents - Semantic Similarity approach
Focuses on semantic similarity-based chunking and sampling for better relevance
"""

import numpy as np
from typing import List, Dict, Optional
import asyncio
import httpx
import json
import os
import tempfile
import traceback
import re
from pathlib import Path
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from starlette.concurrency import run_in_threadpool

# Environment setup
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment")


async def get_embeddings(texts: List[str], model: str = "voyage-3.5-lite") -> List[List[float]]:
    """Get embeddings from Voyage AI API"""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY not found in environment")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": texts,
                "model": model
            }
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


async def calculate_semantic_scores(chunks: List[Document], query: str = None) -> List[float]:
    """Calculate semantic similarity scores for chunks"""

    # If no query provided, use a generic business/document query
    if not query:
        query = "important terms conditions coverage benefits requirements procedures definitions"

    # Prepare texts for embedding
    chunk_texts = []
    for chunk in chunks:
        # Use first 500 chars to avoid token limits while preserving context
        text = chunk.page_content[:500].strip()
        if len(text) < 50:  # If too short, use full text
            text = chunk.page_content.strip()
        chunk_texts.append(text)

    # Get embeddings for all chunks and query
    all_texts = [query] + chunk_texts

    try:
        embeddings = await get_embeddings(all_texts)
        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        # Calculate similarity scores
        scores = []
        for chunk_embedding in chunk_embeddings:
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            scores.append(max(similarity, 0.0))  # Ensure non-negative

        return scores

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        # Fallback to basic text length scoring
        return [min(len(text.split()) / 100, 1.0) for text in chunk_texts]


async def semantic_chunk_sampling(chunks: List[Document], max_chunks: int = 100, query: str = None) -> List[Document]:
    """Smart sampling based on semantic similarity"""

    if len(chunks) <= max_chunks:
        return chunks

    print(f"Calculating semantic scores for {len(chunks)} chunks...")

    # Get semantic similarity scores
    semantic_scores = await calculate_semantic_scores(chunks, query)

    # Combine chunks with their scores
    scored_chunks = list(zip(chunks, semantic_scores))

    # Group by section if available
    section_groups = defaultdict(list)
    standalone_chunks = []

    for chunk, score in scored_chunks:
        section_title = chunk.metadata.get('section_title', '')
        if section_title:
            section_groups[section_title].append((chunk, score))
        else:
            standalone_chunks.append((chunk, score))

    selected_chunks = []

    # Process sections - take top chunks from each section based on semantic score
    if section_groups:
        # Calculate average semantic score per section
        section_avg_scores = []
        for section_title, section_chunk_scores in section_groups.items():
            avg_score = sum(score for _, score in section_chunk_scores) / len(section_chunk_scores)
            section_avg_scores.append((section_title, avg_score, section_chunk_scores))

        # Sort sections by average semantic relevance
        section_avg_scores.sort(key=lambda x: x[1], reverse=True)

        # Distribute chunks across semantically relevant sections
        chunks_per_section = max(2, max_chunks // max(len(section_groups), 1))

        for section_title, avg_score, section_chunk_scores in section_avg_scores:
            # Sort chunks within section by semantic score
            section_chunk_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top semantically similar chunks from this section
            section_limit = min(len(section_chunk_scores), chunks_per_section)
            selected_chunks.extend([chunk for chunk, score in section_chunk_scores[:section_limit]])

            if len(selected_chunks) >= max_chunks:
                break

    # Add standalone chunks based on semantic relevance
    remaining_slots = max_chunks - len(selected_chunks)
    if remaining_slots > 0 and standalone_chunks:
        standalone_chunks.sort(key=lambda x: x[1], reverse=True)
        selected_chunks.extend([chunk for chunk, score in standalone_chunks[:remaining_slots]])

    # Final selection if still over limit - keep highest semantic scores
    if len(selected_chunks) > max_chunks:
        # Re-score if needed
        final_scores = []
        for chunk in selected_chunks:
            # Find the chunk's semantic score from our original scoring
            for orig_chunk, score in scored_chunks:
                if orig_chunk == chunk:
                    final_scores.append((chunk, score))
                    break

        final_scores.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk for chunk, score in final_scores[:max_chunks]]

    print(f"Selected {len(selected_chunks)} chunks based on semantic similarity")
    return selected_chunks


class InsuranceSectionSplitter:
    """Custom splitter that understands insurance document structure"""

    # Common insurance section patterns
    SECTION_PATTERNS = [
        r'^\s*(?:SECTION\s+)?([IVXLCDM]+|\d+)\.\s*([A-Z][A-Z\s&-]+)(?:\s*[-–—]\s*)?',  # "1. DEFINITIONS" or "I. COVERAGE"
        r'^\s*([A-Z][A-Z\s&-]{10,})\s*$',  # All caps headers like "GENERAL CONDITIONS"
        r'^\s*Part\s+([IVXLCDM]+|\d+)[\s:]?\s*([A-Z][A-Za-z\s&-]+)',  # "Part I: Liability Coverage"
        r'^\s*Article\s+(\d+)[\s:]?\s*([A-Z][A-Za-z\s&-]+)',  # "Article 1: Definitions"
        r'^\s*Chapter\s+(\d+)[\s:]?\s*([A-Z][A-Za-z\s&-]+)',  # "Chapter 1: General Provisions"
    ]

    # Critical insurance keywords that should stay together
    CRITICAL_KEYWORDS = [
        "coverage", "exclusion", "sum insured", "premium", "claim",
        "deductible", "co-payment", "reimbursement", "cashless",
        "pre-existing disease", "waiting period", "grace period",
        "policy", "insurer", "insured", "policyholder", "hospitalization"
    ]

    def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving section boundaries"""
        all_chunks = []

        for doc in documents:
            chunks = self._split_single_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def _split_single_document(self, document: Document) -> List[Document]:
        """Split a single document preserving insurance sections"""
        text = document.page_content
        sections = self._identify_sections(text)

        if not sections:
            # Fallback to regular splitting if no sections found
            return self._fallback_split(document)

        chunks = []
        for section_title, section_text, start_pos in sections:
            section_chunks = self._split_section(
                section_text,
                section_title,
                document.metadata,
                start_pos
            )
            chunks.extend(section_chunks)

        return chunks

    def _identify_sections(self, text: str) -> List[tuple]:
        """Identify logical sections in insurance documents"""
        sections = []
        lines = text.split('\n')
        current_section = []
        current_title = ""
        section_start = 0

        for i, line in enumerate(lines):
            # Check if this line is a section header
            is_header = False
            title = ""

            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line.strip())
                if match:
                    is_header = True
                    title = line.strip()
                    break

            if is_header and current_section:
                # Save previous section
                section_text = '\n'.join(current_section)
                if len(section_text.strip()) > 100:  # Only keep substantial sections
                    sections.append((current_title, section_text, section_start))

                # Start new section
                current_section = [line]
                current_title = title
                section_start = i
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            if len(section_text.strip()) > 100:
                sections.append((current_title, section_text, section_start))

        return sections

    def _split_section(self, section_text: str, section_title: str,
                      base_metadata: Dict, start_pos: int) -> List[Document]:
        """Split a section while keeping related content together"""

        # For small sections, keep them whole
        if len(section_text) <= self.chunk_size * 1.5:
            metadata = base_metadata.copy()
            metadata.update({
                'section_title': section_title,
                'section_start': start_pos,
                'chunk_type': 'complete_section'
            })
            # Add section title as header for better LLM context
            enhanced_content = f"{section_title}\n\n{section_text}" if section_title else section_text
            return [Document(page_content=enhanced_content, metadata=metadata)]

        # Split larger sections using insurance-aware approach
        chunks = []

        # Use custom separators that respect insurance document structure
        separators = [
            r'\n\s*(?:\([a-z]\)|\([0-9]+\)|\([IVXLCDM]+\))',  # Subsection markers like (a), (1), (i)
            r'\n\s*[a-z]\.\s+',  # List items like "a. "
            r'\n\s*\d+\.\s+',    # Numbered items like "1. "
            r'(?<=[.!?])\s+(?=[A-Z])',  # Sentence boundaries
            r'\n\n+',            # Paragraph breaks
            r'\n',               # Line breaks
            r'\.\s+',            # Sentence endings
            r'\s+'               # Word boundaries (last resort)
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=True,
        )

        section_chunks = splitter.split_text(section_text)

        for i, chunk_text in enumerate(section_chunks):
            # Ensure chunk contains enough context
            if self._has_sufficient_context(chunk_text):
                metadata = base_metadata.copy()
                metadata.update({
                    'section_title': section_title,
                    'section_start': start_pos,
                    'chunk_index': i,
                    'chunk_type': 'section_part'
                })
                # Add section title as header for better LLM context
                enhanced_content = f"{section_title}\n\n{chunk_text}" if section_title else chunk_text
                chunks.append(Document(page_content=enhanced_content, metadata=metadata))

        return chunks

    def _has_sufficient_context(self, text: str) -> bool:
        """Very lenient context check for semantic approach"""
        # Let semantic similarity handle quality - just check it's not empty
        return len(text.strip()) > 10 and len(text.split()) > 2

    def _fallback_split(self, document: Document) -> List[Document]:
        """Fallback splitting when no sections are detected"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                r'\n\n\n+',
                r'\n\n',
                r'(?<=[.!?])\s+(?=[A-Z])',
                r'\n',
                r'\.\s+',
                r'\s+'
            ],
            length_function=len,
            is_separator_regex=True,
        )

        chunks = splitter.split_documents([document])

        # Filter and enhance metadata
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            if self._has_sufficient_context(chunk.page_content):
                chunk.metadata.update({
                    'chunk_index': i,
                    'chunk_type': 'fallback_split'
                })
                filtered_chunks.append(chunk)

        return filtered_chunks


def get_smart_splitter(num_pages: int) -> InsuranceSectionSplitter:
    """Get section-aware splitter with page-based sizing"""
    if num_pages > 300:
        chunk_size, chunk_overlap = 3500, 500
    elif num_pages > 150:
        chunk_size, chunk_overlap = 3000, 400
    elif num_pages > 75:
        chunk_size, chunk_overlap = 2500, 350
    elif num_pages > 40:
        chunk_size, chunk_overlap = 2000, 300
    elif num_pages > 20:
        chunk_size, chunk_overlap = 1800, 250
    else:
        chunk_size, chunk_overlap = 1500, 200

    return InsuranceSectionSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def has_sufficient_context_semantic(text: str) -> bool:
    """Very lenient context check - let semantic similarity handle quality"""
    return len(text.strip()) > 10 and len(text.split()) > 2


def _is_likely_toc_or_header(text: str) -> bool:
    """Detect and filter table of contents or header/footer pages"""
    text_lower = text.lower().strip()

    # Check for table of contents indicators
    toc_indicators = [
        'table of contents', 'contents', 'index',
        'page', 'section', '..........', '......'
    ]

    # Check if it's mostly page numbers and dots
    lines = text.split('\n')
    numeric_lines = sum(1 for line in lines if re.search(r'\d+\s*$', line.strip()))

    if numeric_lines > len(lines) * 0.5:  # More than 50% lines end with numbers
        return True

    # Check for TOC keywords
    if any(indicator in text_lower for indicator in toc_indicators):
        return True

    # Very short pages are likely headers/footers
    if len(text.split()) < 20:
        return True

    return False


async def load_insurance_pdf_semantic(path_or_url: str, max_chunks: int = 100, query: str = None) -> List[Document]:
    """Enhanced PDF loading with semantic similarity-based chunking"""
    temp_download = False

    if path_or_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)
        ) as client:
            response = await client.get(path_or_url, follow_redirects=True)
            response.raise_for_status()

            file_path = Path(tempfile.mkstemp(suffix=".pdf")[1])
            with open(file_path, "wb") as f:
                f.write(response.content)
            temp_download = True
            print(f"Downloaded PDF to temporary file: {file_path}")
    else:
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_and_split():
        try:
            print(f"Loading PDF from: {file_path}")

            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()

            print(f"Raw documents loaded: {len(raw_docs)}")

            if not raw_docs:
                raise Exception("PyPDFLoader returned no documents")

            # Very minimal filtering - only remove completely empty pages
            filtered_docs = [
                doc for doc in raw_docs
                if len(doc.page_content.strip()) > 20  # Very lenient - just non-empty
                and doc.page_content.strip().count(' ') > 3  # At least a few words
            ]

            print(f"Documents after minimal filtering: {len(filtered_docs)}")

            # If still no docs, use all raw docs (let semantic similarity handle quality)
            if not filtered_docs:
                print("Using all raw documents - semantic similarity will handle quality")
                filtered_docs = raw_docs

            # Use section-aware splitter for better chunking
            splitter = get_smart_splitter(len(filtered_docs))
            split_docs = splitter.split_documents(filtered_docs)

            print(f"Documents after splitting: {len(split_docs)}")

            # Additional filtering for semantic approach
            semantic_filtered = [
                doc for doc in split_docs
                if has_sufficient_context_semantic(doc.page_content)
            ]

            print(f"Documents after semantic filtering: {len(semantic_filtered)}")
            return semantic_filtered

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error processing PDF {file_path}: {e}")
            print(f"Full traceback:\n{error_details}")
            raise Exception(f"PDF processing failed for {file_path}: {str(e)}") from e

    split_docs = await run_in_threadpool(load_and_split)

    if temp_download and file_path.exists():
        try:
            os.remove(file_path)
        except:
            pass

    if not split_docs:
        raise Exception("No content extracted from PDF")

    # Apply semantic similarity-based sampling
    final_docs = await semantic_chunk_sampling(split_docs, max_chunks, query)

    return final_docs


# Main API functions with consistent naming
async def load_pdf_ultra_fast(path_or_url: str, query: str = None) -> List[Document]:
    """Ultra fast PDF loading with semantic similarity - main API function"""
    return await load_insurance_pdf_semantic(path_or_url, max_chunks=100, query=query)

async def load_pdf_semantic(path_or_url: str, query: str = None) -> List[Document]:
    """Semantic similarity-based PDF loading - alternative API"""
    return await load_insurance_pdf_semantic(path_or_url, max_chunks=100, query=query)


def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass