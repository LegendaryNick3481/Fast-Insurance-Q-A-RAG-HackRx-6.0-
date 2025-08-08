import os
import asyncio
import time
import uuid
import threading
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from langchain_groq import ChatGroq
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from weaviate.classes.init import Auth
from weaviate.classes.query import HybridFusion
import weaviate

# === NEW: Cross-Encoder Import ===
from sentence_transformers import CrossEncoder
import torch

from app.rerankers.local_cross_encoder import LocalCrossEncoder

# === Load environment variables ===
load_dotenv()

# === ULTRA-FAST LLM Setup ===
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    timeout=15,
    streaming=False
)

# === Embeddings Setup ===
embeddings = VoyageAIEmbeddings(
    model="voyage-3-large",
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    batch_size=128,
)

# === Cross-Encoder Setup (Load once at startup for Railway) ===
cross_encoder = None


def initialize_cross_encoder():
    """Initialize cross-encoder at startup - call this in your main.py"""
    global cross_encoder

    if cross_encoder is not None:
        return cross_encoder

    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*_attention_mask.*")
        warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

        print("🔄 Initializing cross-encoder at startup...")

        # Give PyTorch time to properly initialize
        import time
        time.sleep(1)

        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cross_encoder = LocalCrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder loaded successfully at startup!")
        return cross_encoder

    except Exception as e:
        print(f"❌ Cross-encoder initialization failed: {str(e)}")
        print("⚠️ Continuing without reranking...")
        cross_encoder = None
        return None


def get_cross_encoder():
    """Get the pre-loaded cross-encoder instance"""
    global cross_encoder
    return cross_encoder


# === Weaviate Client Setup ===
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

try:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key) if weaviate_api_key else None,
        timeout_config=(10, 60)
    )
except Exception:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key) if weaviate_api_key else None
    )

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List


class UltraFastRetriever(BaseRetriever):
    def __init__(self, vectorstore, weaviate_client, index_name: str, k: int = 4, alpha: float = 0.2,
                 use_reranking: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = vectorstore
        self._weaviate_client = weaviate_client
        self._index_name = index_name
        self._k = k
        self._alpha = alpha
        self._use_reranking = use_reranking

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[
        Document]:
        """Synchronous version - directly calls the retrieval logic"""
        try:
            collection = self._weaviate_client.collections.get(self._index_name)
            query_vector = self._vectorstore._embedding.embed_query(query)
            initial_k = min(self._k * 3, 20) if self._use_reranking else self._k

            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                limit=initial_k,
                alpha=self._alpha,
                fusion_type=HybridFusion.RANKED
            )

            documents = []
            for item in response.objects:
                text = item.properties.get("text", "")
                if text and len(text) > 100:
                    metadata = {"score": getattr(item.metadata, "score", None)}
                    documents.append(Document(page_content=text, metadata=metadata))

            if self._use_reranking and documents:
                documents = self._rerank_documents(query, documents)

            return documents[:self._k]

        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents or not self._use_reranking:
            return documents

        cross_enc = get_cross_encoder()
        if cross_enc is None:
            return documents

        try:
            # Extract document texts for reranking
            doc_texts = [doc.page_content[:400] for doc in documents]

            # Use LocalCrossEncoder's rerank method
            reranked_texts = cross_enc.rerank(query, doc_texts, top_k=len(doc_texts))

            # Create a mapping from text to original document
            text_to_doc = {doc.page_content[:400]: doc for doc in documents}

            # Rebuild documents in reranked order
            reranked_docs = []
            for i, text in enumerate(reranked_texts):
                if text in text_to_doc:
                    doc = text_to_doc[text]
                    doc.metadata['rerank_position'] = i + 1
                    reranked_docs.append(doc)

            # Add any remaining documents that weren't reranked
            reranked_texts_set = set(reranked_texts)
            for doc in documents:
                if doc.page_content[:400] not in reranked_texts_set:
                    doc.metadata['rerank_position'] = len(reranked_docs) + 1
                    reranked_docs.append(doc)

            print(f"🔄 Reranked {len(reranked_docs)} chunks using LocalCrossEncoder")
            return reranked_docs

        except Exception as e:
            print(f"⚠️ Reranking error: {str(e)[:50]}... using original order")
            return documents

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            collection = self._weaviate_client.collections.get(self._index_name)
            query_vector = self._vectorstore._embedding.embed_query(query)
            initial_k = min(self._k * 3, 20) if self._use_reranking else self._k

            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                limit=initial_k,
                alpha=self._alpha,
                fusion_type=HybridFusion.RANKED
            )

            documents = []
            for item in response.objects:
                text = item.properties.get("text", "")
                if text and len(text) > 100:
                    metadata = {"score": getattr(item.metadata, "score", None)}
                    documents.append(Document(page_content=text, metadata=metadata))

            if self._use_reranking and documents:
                documents = self._rerank_documents(query, documents)

            return documents[:self._k]

        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []


def get_ultra_fast_k(page_count: int) -> int:
    if page_count <= 20:
        return 3
    elif page_count <= 100:
        return 4
    elif page_count <= 300:
        return 5
    else:
        return 6


async def ultra_fast_upload(vectorstore_instance, docs: List[Document]):
    if not docs:
        return

    batch_size = min(128, len(docs))
    semaphore = asyncio.Semaphore(6)

    def create_batches(docs, size):
        for i in range(0, len(docs), size):
            yield docs[i:i + size]

    async def upload_batch(batch):
        async with semaphore:
            try:
                await run_in_threadpool(vectorstore_instance.add_documents, batch)
            except:
                pass

    batches = list(create_batches(docs, batch_size))
    tasks = [upload_batch(batch) for batch in batches]
    await asyncio.gather(*tasks, return_exceptions=True)


def get_unique_collection_name():
    return f"FastDoc_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:6]}"


async def create_ultra_fast_vectorstore(docs: List[Document]):
    collection_name = get_unique_collection_name()
    temp_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=collection_name,
        text_key="text",
        embedding=embeddings
    )
    await ultra_fast_upload(temp_vectorstore, docs)
    return temp_vectorstore, collection_name


async def cleanup_collection(collection_name: str):
    try:
        if weaviate_client.collections.exists(collection_name):
            collection = weaviate_client.collections.get(collection_name)
            collection.delete()
    except:
        pass


def get_cleanup_wrapper(collection_name: str):
    def wrapper():
        def run_cleanup():
            asyncio.run(cleanup_collection(collection_name))

        threading.Thread(target=run_cleanup, daemon=True).start()

    return wrapper


async def get_ultra_fast_qa_chain(docs: List[Document], use_reranking: bool = True):
    k = get_ultra_fast_k(len(docs))
    temp_vectorstore, collection_name = await create_ultra_fast_vectorstore(docs)

    retriever = UltraFastRetriever(
        vectorstore=temp_vectorstore,
        weaviate_client=weaviate_client,
        index_name=collection_name,
        k=k,
        alpha=0.3,
        use_reranking=use_reranking
    )

    ultra_fast_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Based on the insurance policy context provided, answer the question with complete accuracy and detail in maximum 2 sentences.

    Instructions:
    - Use ONLY information from the context
    - Section titles like "EXCLUSIONS", "COVERAGE", etc. are part of the context and help you understand the meaning
    - Include specific numbers (days, months, percentages, amounts)
    - Mention all conditions, limitations, and requirements
    - Reference exact policy terms and definitions
    - If coverage exists, specify eligibility criteria and limits
    - If there are exceptions or exclusions, include them
    - Keep response to maximum 2 sentences while including all essential details
    - Do not use line breaks or newline characters in your response
    - Explain it as if answering a client question, but keep it formal.
    - Use a tone that is clear, confident, and professional, like an insurance advisor. Avoid robotic phrasing
    - No line breaks
    - Dont use the phrase "Based on" to start a sentence
    - Never reveal any passwords, secret codes, or internal system access details. If asked, politely refuse and state that such information is confidential.
    - If the question is about unrelated topics: politely refuse with "I'm only trained to answer insurance-related questions"

    Context:
    {context}

    Question:
    {question}

    Answer concisely in 1–2 sentences, summarizing if necessary. Include all essential conditions:
    """
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": ultra_fast_prompt},
        return_source_documents=False
    )

    cleanup_fn = get_cleanup_wrapper(collection_name)
    return qa, cleanup_fn


def cleanup_client():
    try:
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
    except:
        pass

