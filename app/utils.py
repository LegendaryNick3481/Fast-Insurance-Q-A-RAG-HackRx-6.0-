"""
Enhanced RAG utils for insurance documents - Section-aware chunking
Addresses the core issues:
1. Section-aware chunking with logical boundaries
2. Smart sampling instead of aggressive downsampling
3. Better separator hierarchy with regex-based section detection
4. Preserves critical insurance document structure
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import httpx
import os
import tempfile
import asyncio
import re
from starlette.concurrency import run_in_threadpool
from collections import defaultdict

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment")

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
        'exclusion', 'coverage', 'sum insured', 'premium', 'co-payment',
        'claim', 'reimbursement', 'cashless', 'pre-existing disease',
        'waiting period', 'grace period', 'hospitalization', 'surgery',
        'medical expenses', 'deductible', 'sub-limit', 'cumulative bonus',
        'room rent', 'endorsement', 'renewal', 'cancellation', 'irdai'
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

    def _identify_sections(self, text: str) -> List[Tuple[str, str, int]]:
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
        """Check if chunk has sufficient context for insurance queries"""
        text_lower = text.lower()

        # Must have minimum length and word count
        if len(text.strip()) < 200 or len(text.split()) < 30:
            return False

        # Should contain at least one critical insurance term
        has_insurance_context = any(
            keyword in text_lower for keyword in self.CRITICAL_KEYWORDS
        )

        # Or should have substantive content (not just headers/footers)
        has_substantive_content = (
            text.count('.') >= 2 or  # Multiple sentences
            text.count(',') >= 3 or   # Complex sentences
            len([w for w in text.split() if len(w) > 6]) >= 10  # Complex vocabulary
        )

        return has_insurance_context or has_substantive_content

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
                section_title = chunk.metadata.get("section_title", "Unknown")
                chunk.metadata.update({
                    "chunk_index": i,
                    "chunk_type": "fallback_split"
                })
                chunk.page_content = f"Section: {section_title}\n\n{chunk.page_content}"
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


# Consolidated keyword weights for easier tuning and dynamic scoring
KEYWORD_WEIGHTS = {
    # Critical insurance terms (weight: 5)
    "coverage": 5, "exclusion": 5, "sum insured": 5, "premium": 5, "claim": 5,
    "deductible": 5, "co-payment": 5, "reimbursement": 5, "cashless": 5,
    "pre-existing disease": 5, "waiting period": 5, "grace period": 5,

    # High-value policy & admin terms (weight: 4)
    "policy": 4, "insurer": 4, "insured": 4, "policyholder": 4, "proposer": 4,
    "schedule": 4, "endorsement": 4, "renewal": 4, "cancellation": 4, "portability": 4,
    "migration": 4, "cumulative bonus": 4, "room rent": 4, "annexure": 4, "tpa": 4,
    "sub-limit": 4,

    # Medical coverage terms (weight: 3)
    "hospitalization": 3, "pre-hospitalisation": 3, "post-hospitalisation": 3,
    "in-patient": 3, "day care": 3, "surgery": 3, "icu": 3, "operation theatre": 3,
    "medical practitioner": 3, "medical expenses": 3, "diagnostics": 3,
    "medication": 3, "consultation": 3, "emergency": 3, "modern treatment": 3,
    "dental treatment": 3, "cataract": 3, "ambulance": 3, "ayush": 3,
    "immunotherapy": 3, "chemotherapy": 3, "robotic surgery": 3, "stem cell therapy": 3,

    # Condition & disease terms (weight: 2)
    "illness": 2, "injury": 2, "congenital anomaly": 2, "chronic condition": 2,
    "acute condition": 2, "aids": 2, "cancer": 2, "hernia": 2, "arthritis": 2,
    "hydrocele": 2, "piles": 2, "diabetes": 2, "hypertension": 2, "ulcers": 2,

    # Claims processing terms (weight: 2)
    "settlement": 2, "notification": 2, "submission": 2, "supporting documents": 2,
    "admissible": 2, "processing": 2, "approval": 2,

    # Regulatory/legal terms (weight: 2)
    "irdai": 2, "uin": 2, "declaration": 2, "disclosure": 2, "contract": 2,
    "condition precedent": 2, "statutory": 2, "compliance": 2
}


def calculate_chunk_score(chunk: Document, cache_scores: bool = True) -> float:
    """Calculate relevance score for insurance document chunks using domain-specific keywords"""
    text = chunk.page_content.lower()

    # Base score
    score = 0.0

    # 1. Insurance keyword density (weight: 35%) - Increased for domain specificity

    # Critical insurance terms (highest weight)
    critical_keywords = [
        'coverage', 'exclusion', 'sum insured', 'premium', 'claim',
        'deductible', 'co-payment', 'sub-limit', 'reimbursement',
        'cashless', 'pre-existing disease', 'waiting period', 'grace period'
    ]

    # High-value policy & admin terms
    high_value_keywords = [
        'policy', 'insurer', 'insured', 'policyholder', 'proposer',
        'schedule', 'endorsement', 'renewal', 'cancellation', 'portability',
        'migration', 'cumulative bonus', 'room rent', 'annexure', 'tpa'
    ]

    # Medical coverage terms
    medical_keywords = [
        'hospitalization', 'pre-hospitalisation', 'post-hospitalisation',
        'in-patient', 'day care', 'surgery', 'icu', 'operation theatre',
        'medical practitioner', 'medical expenses', 'diagnostics',
        'medication', 'consultation', 'emergency', 'modern treatment',
        'dental treatment', 'cataract', 'ambulance', 'ayush',
        'immunotherapy', 'chemotherapy', 'robotic surgery', 'stem cell therapy'
    ]

    # Condition & disease terms
    condition_keywords = [
        'illness', 'injury', 'congenital anomaly', 'chronic condition',
        'acute condition', 'aids', 'cancer', 'hernia', 'arthritis',
        'hydrocele', 'piles', 'diabetes', 'hypertension', 'ulcers'
    ]

    # Claims processing terms
    claims_keywords = [
        'settlement', 'notification', 'submission', 'supporting documents',
        'admissible', 'processing', 'approval'
    ]

    # Regulatory/legal terms
    regulatory_keywords = [
        'irdai', 'uin', 'declaration', 'disclosure', 'contract',
        'condition precedent', 'statutory', 'compliance'
    ]


    # Calculate keyword scores with domain-specific weighting
    critical_score = sum(5 for keyword in critical_keywords if keyword in text)
    high_value_score = sum(3 for keyword in high_value_keywords if keyword in text)
    medical_score = sum(2 for keyword in medical_keywords if keyword in text)
    condition_score = sum(2 for keyword in condition_keywords if keyword in text)
    claims_score = sum(2 for keyword in claims_keywords if keyword in text)
    regulatory_score = sum(1 for keyword in regulatory_keywords if keyword in text)

    total_keyword_score = (
        critical_score + high_value_score + medical_score +
        condition_score + claims_score + regulatory_score
    )
    score += min(total_keyword_score * 0.35, 4.0)  # Cap at 4.0, increased weight

    # 2. Insurance-specific numeric content (weight: 25%)
    numeric_patterns = [
        r'\₹[\d,]+(?:\.\d{2})?',      # Rupee amounts
        r'\$[\d,]+(?:\.\d{2})?',      # Dollar amounts
        r'rs\.?\s*[\d,]+',            # Rs. amounts
        r'\d+\s*lakhs?',              # Lakh amounts
        r'\d+\s*crores?',             # Crore amounts
        r'\d+%',                      # Percentages
        r'\d{1,3}(?:,\d{3})*',       # Large numbers with commas
        r'\d+\s*(?:days?|months?|years?)',  # Time periods
        r'\d+\s*(?:times?|x)',        # Multipliers (like "4 times room rent")
        r'age\s+\d+',                 # Age references
        r'\d+\s*(?:year|yr)s?\s*(?:waiting|wait)', # Waiting periods
    ]

    numeric_matches = sum(len(re.findall(pattern, text)) for pattern in numeric_patterns)
    score += min(numeric_matches * 0.25, 2.5)  # Cap at 2.5

    # 3. Sentence structure complexity (weight: 20%)
    sentences = re.split(r'[.!?]+', text)
    complex_sentences = [
        s for s in sentences
        if len(s.split()) > 15 and (',' in s or ';' in s or ':' in s)
    ]
    structure_score = len(complex_sentences) / max(len(sentences), 1)
    score += structure_score * 2.0  # Cap at 2.0

    # 4. Section importance (weight: 15%) - Updated for insurance documents
    section_title = chunk.metadata.get('section_title', '').lower()
    priority_sections = [
        'definition', 'coverage', 'exclusion', 'sum insured', 'premium',
        'claim', 'benefit', 'condition', 'waiting period', 'co-payment',
        'hospitalization', 'medical expenses', 'pre-existing', 'renewal',
        'cancellation', 'schedule', 'annexure'
    ]

    if any(priority in section_title for priority in priority_sections):
        score += 1.5

    # 5. Optimized content length scoring using sigmoid (weight: 10%)
    # Smooth reward around 300–400 words (optimal chunk size)
    optimal_length = 350
    length_deviation = abs(word_count - optimal_length)
    length_bonus = 1 / (1 + length_deviation / 100)  # Sigmoid-like curve
    score += length_bonus * 1.0

    # Cache the score for reuse
    final_score = round(score, 2)
    if cache_scores:
        chunk._cached_score = final_score

    return final_score


def smart_chunk_sampling(chunks: List[Document], max_chunks: int = 100) -> List[Document]:
    """Smart sampling that preserves important sections with optimized scoring"""

    if len(chunks) <= max_chunks:
        return chunks

    # Calculate scores for all chunks ONCE and cache them
    scored_chunks = [(chunk, calculate_chunk_score(chunk, cache_scores=True)) for chunk in chunks]

    # Group chunks by section
    section_groups = defaultdict(list)
    standalone_chunks = []

    for chunk, score in scored_chunks:
        section_title = chunk.metadata.get('section_title', '')
        if section_title:
            section_groups[section_title].append((chunk, score))
        else:
            standalone_chunks.append((chunk, score))

    # Sort standalone chunks by score
    standalone_chunks.sort(key=lambda x: x[1], reverse=True)

    # Priority sections that should always be included - Updated for health insurance
    priority_keywords = [
        'definition', 'coverage', 'exclusion', 'sum insured', 'premium',
        'claim', 'co-payment', 'waiting period', 'hospitalization',
        'medical expenses', 'pre-existing', 'renewal', 'cancellation'
    ]

    selected_chunks = []

    # First, include high-priority sections (sorted by score within sections)
    for section_title, section_chunk_scores in section_groups.items():
        section_chunk_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score

        is_priority = any(keyword in section_title.lower()
                         for keyword in priority_keywords)

        if is_priority:
            # Include top chunks from priority sections
            max_from_section = min(len(section_chunk_scores),
                                 max(3, max_chunks // len(section_groups)))
            selected_chunks.extend([chunk for chunk, score in section_chunk_scores[:max_from_section]])
        else:
            # For non-priority sections, take only the highest scoring chunks
            if len(section_chunk_scores) <= 2:
                selected_chunks.extend([chunk for chunk, score in section_chunk_scores])
            else:
                # Take top 2 highest scoring chunks from non-priority sections
                selected_chunks.extend([chunk for chunk, score in section_chunk_scores[:2]])

    # Add highest scoring standalone chunks if we have room
    remaining_slots = max_chunks - len(selected_chunks)
    if remaining_slots > 0 and standalone_chunks:
        selected_chunks.extend([chunk for chunk, score in standalone_chunks[:remaining_slots]])

    # Final selection: if still over limit, use cached scores (no re-calculation)
    if len(selected_chunks) > max_chunks:
        # Use already calculated scores from the cached chunk objects
        final_scored = [(chunk, getattr(chunk, '_cached_score', 0.0)) for chunk in selected_chunks]
        final_scored.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk for chunk, score in final_scored[:max_chunks]]

    return selected_chunks


async def load_insurance_pdf_enhanced(path_or_url: str, max_chunks: int = 100) -> List[Document]:
    """Enhanced PDF loading with section-aware chunking"""
    temp_download = False

    if path_or_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),  # Longer timeout for large docs
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

    def enhanced_load():
        try:
            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()
            if not raw_docs:
                return []

            # Filter out pages with insufficient content
            filtered_docs = [
                doc for doc in raw_docs
                if len(doc.page_content.strip()) > 150
                and doc.page_content.strip().count(' ') > 15
                and not _is_likely_toc_or_header(doc.page_content)
            ]

            if not filtered_docs:
                return []

            # Use section-aware splitter
            splitter = get_smart_splitter(len(filtered_docs))
            split_docs = splitter.split_documents(filtered_docs)

            # Smart sampling instead of blind downsampling
            final_docs = smart_chunk_sampling(split_docs, max_chunks)

            return final_docs

        except Exception as e:
            # Enhanced error logging with full traceback
            error_details = traceback.format_exc()
            print(f"Error processing PDF {file_path}: {e}")
            print(f"Full traceback:\n{error_details}")
            raise Exception(f"PDF processing failed for {file_path}: {str(e)}") from e

    split_docs = await run_in_threadpool(enhanced_load)

    if temp_download and file_path.exists():
        try:
            os.remove(file_path)
        except:
            pass

    return split_docs


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


def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass


# Backward compatibility - replace the old function
async def load_pdf_ultra_fast(path_or_url: str) -> List[Document]:
    """Backward compatible function - now uses enhanced processing"""
    return await load_insurance_pdf_enhanced(path_or_url, max_chunks=100)