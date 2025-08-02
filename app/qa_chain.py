from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Weaviate
from langchain_community.embeddings import VoyageAIEmbeddings
from langchain_community.retrievers import VoyageAIRerankRetriever
from weaviate.client import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
import os


def get_qa_chain(docs):
    # Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url:
        raise ValueError("WEAVIATE_URL not set.")

    connection_params = ConnectionParams.from_params(
        http_host=weaviate_url.replace("https://", "").replace("http://", ""),
        http_secure=True
    )
    auth = AuthApiKey(weaviate_api_key) if weaviate_api_key else None
    client = WeaviateClient(connection_params=connection_params, auth=auth)

    # VoyageAI Embeddings
    embeddings = VoyageAIEmbeddings(
        model="voyage-3-large",
        voyage_api_key=os.getenv("VOYAGE_API_KEY")
    )

    # Add docs to Weaviate
    vectorstore = Weaviate(
        client=client,
        index_name="Document",
        text_key="text",
        embedding=embeddings,
        create_schema_if_missing=True
    )
    vectorstore.add_documents(docs)

    # Hybrid Retriever from Weaviate
    hybrid_retriever = vectorstore.as_retriever(
        search_type="hybrid",
        search_kwargs={"k": 8, "alpha": 0.5}
    )

    # Wrap with VoyageAI Reranker
    reranker = VoyageAIRerankRetriever.from_retriever(
        base_retriever=hybrid_retriever,
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        model="rerank-2",  # or rerank-2-lite
        top_n=4
    )

    # LLM via Groq
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

    # Final RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=reranker,
        return_source_documents=True
    )

    return qa
