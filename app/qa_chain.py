import os
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
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain.retrievers import ContextualCompressionRetriever

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import HybridFusion

# === Load .env ===
load_dotenv()

# === LLM Setup ===
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0,
)

# === Embedding ===
embeddings = VoyageAIEmbeddings(
    model="voyage-3-large",
    voyage_api_key=os.getenv("VOYAGE_API_KEY")
)

# === Weaviate client ===
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key) if weaviate_api_key else None
)

# === Vector Store ===
vectorstore = WeaviateVectorStore(
    client=weaviate_client,
    index_name="Document",
    text_key="text",
    embedding=embeddings
)

# === Custom Hybrid Retriever ===
class WeaviateHybridRetriever(BaseRetriever):
    def __init__(self, vectorstore, weaviate_client, index_name: str, k: int = 8, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = vectorstore
        self._weaviate_client = weaviate_client
        self._index_name = index_name  # Store it separately
        self._k = k
        self._alpha = alpha

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            # Use the stored index_name instead of trying to access it from vectorstore
            collection = self._weaviate_client.collections.get(self._index_name)

            # Access the embedding properly
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
                metadata = {k: v for k, v in item.properties.items() if k != "text"}
                metadata["score"] = getattr(item.metadata, "score", None)
                metadata["uuid"] = str(item.uuid)
                documents.append(Document(page_content=text, metadata=metadata))

            return documents

        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

def get_dynamic_k(page_count: int) -> int:
    if page_count <= 5:
        return 4
    elif page_count <= 20:
        return 6
    elif page_count <= 50:
        return 8
    elif page_count <= 100:
        return 10
    elif page_count <= 200:
        return 12
    elif page_count <= 350:
        return 16
    elif page_count <= 500:
        return 20
    else:
        return 24

# === QA Chain Factory ===
async def get_qa_chain(docs: List[Document]):
    k = get_dynamic_k(len(docs))

    await run_in_threadpool(vectorstore.add_documents, docs)

    hybrid_retriever = WeaviateHybridRetriever(
        vectorstore=vectorstore,
        weaviate_client=weaviate_client,
        index_name="Document",  # ✅ explicitly passed
        k=k,
        alpha=0.3
    )

    reranker = VoyageAIRerank(
        model="rerank-2",
        voyageai_api_key=os.getenv("VOYAGE_API_KEY"),
        top_k=  min(k,8)
    )

    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever
    )

    map_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are analyzing an official insurance policy document. Your task is to extract **only directly stated facts**, including **verbatim clauses**, **defined terms**, and **explicitly listed conditions or exclusions** that relate to the question.

    Focus especially on:
    - Waiting periods (e.g., in months or years, and when they begin)
    - Eligibility criteria (e.g., age, gender, prior coverage, continuous coverage requirements)
    - Policy limits (e.g., number of claims, amount caps, frequency restrictions)
    - Coverage details and benefits (what is covered, excluded, or conditionally covered)
    - Definitions of terms (e.g., hospital, medical practitioner, pre-existing diseases)
    - Legal or regulatory compliance clauses (e.g., specific acts or IRDAI references)

    Extract specific numbers, percentages, time periods, and qualifying conditions exactly as stated.

    If the document does not mention anything relevant, write exactly: "No relevant information found."

    Context:
    {context}

    Question: {question}

    Extracted facts (verbatim or minimally rephrased):
    """
    )

    reduce_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""
    Using only the extracted facts provided below, write a direct, complete answer to the question. 

    Your answer must:
    - Be a single, comprehensive paragraph that directly answers the question
    - Include all relevant specific details: exact time periods, percentages, amounts, conditions, and qualifying criteria
    - Mention specific acts, regulations, or compliance requirements when relevant
    - Use precise language from the policy document
    - Start with a clear statement (Yes/No when applicable) followed by specific details
    - If multiple conditions apply, list them clearly within the paragraph

    If the answer is not found in the facts, write: "Not mentioned in the policy document."

    Facts:
    {summaries}

    Question: {question}

    Answer:
    """
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=rerank_retriever,
        chain_type_kwargs={
            "question_prompt": map_prompt,
            "combine_prompt": reduce_prompt
        },
        return_source_documents=False
    )

    return qa

# === Optional Cleanup ===
def cleanup_client():
    if weaviate_client.is_connected():
        weaviate_client.close()
