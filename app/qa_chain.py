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

# === Custom Retriever ===
class UltraFastRetriever(BaseRetriever):
    def __init__(self, vectorstore, weaviate_client, index_name: str, k: int = 4, alpha: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = vectorstore
        self._weaviate_client = weaviate_client
        self._index_name = index_name
        self._k = k
        self._alpha = alpha

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            collection = self._weaviate_client.collections.get(self._index_name)
            query_vector = self._vectorstore._embedding.embed_query(query)
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                limit=self._k,
                alpha=self._alpha,
                fusion_type=HybridFusion.RANKED
            )
            documents = []
            for item in response.objects:
                text = item.properties.get("text", "")
                if text and len(text) > 100:
                    metadata = {"score": getattr(item.metadata, "score", None)}
                    documents.append(Document(page_content=text, metadata=metadata))
            return documents[:self._k]
        except:
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

# === Upload and Vectorstore Utilities ===
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

# === Async Cleanup Task ===
async def cleanup_collection(collection_name: str):
    try:
        if weaviate_client.collections.exists(collection_name):
            collection = weaviate_client.collections.get(collection_name)
            collection.delete()
    except:
        pass

# === Sync Wrapper for Background Cleanup ===
def get_cleanup_wrapper(collection_name: str):
    def wrapper():
        def run_cleanup():
            asyncio.run(cleanup_collection(collection_name))
        threading.Thread(target=run_cleanup, daemon=True).start()
    return wrapper

# === QA Chain ===
async def get_ultra_fast_qa_chain(docs: List[Document]):
    k = get_ultra_fast_k(len(docs))
    temp_vectorstore, collection_name = await create_ultra_fast_vectorstore(docs)

    retriever = UltraFastRetriever(
        vectorstore=temp_vectorstore,
        weaviate_client=weaviate_client,
        index_name=collection_name,
        k=k,
        alpha=0.3
    )

    ultra_fast_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on the insurance policy context provided, answer the question with complete accuracy and detail in maximum 2 sentences.

Instructions:
- Use ONLY information from the context
- Include specific numbers (days, months, percentages, amounts)
- Mention all conditions, limitations, and requirements
- Reference exact policy terms and definitions
- If coverage exists, specify eligibility criteria and limits
- If there are exceptions or exclusions, include them
- Keep response to maximum 2 sentences while including all essential details
- Do not use line breaks or newline characters in your response
- Make it sound as a Human

Context: {context}

Question: {question}

Detailed Answer (maximum 2 sentences, no line breaks):"""
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

# === Cleanup Weaviate Client on Shutdown ===
def cleanup_client():
    try:
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
    except:
        pass
