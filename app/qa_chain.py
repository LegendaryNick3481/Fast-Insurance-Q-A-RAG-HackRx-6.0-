from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Weaviate
from langchain_community.embeddings import VoyageAIEmbeddings
from weaviate.client import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
import os


def get_qa_chain(docs):
    # Step 1: Connect to Weaviate (v4)
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url:
        raise ValueError("WEAVIATE_URL not set in environment.")

    try:
        connection_params = ConnectionParams.from_params(
            http_host=weaviate_url.replace("https://", "").replace("http://", ""),
            http_secure=True
        )
        auth = AuthApiKey(weaviate_api_key) if weaviate_api_key else None
        client = WeaviateClient(
            connection_params=connection_params,
            auth=auth
        )
    except Exception as e:
        raise

    # Step 2: Create VoyageAI Embeddings
    try:
        embeddings = VoyageAIEmbeddings(
            model="voyage-3-large",
            voyage_api_key=os.getenv("VOYAGE_API_KEY")
        )
    except Exception as e:
        raise

    # Step 3: Upload documents to Weaviate
    try:
        vectorstore = Weaviate(
            client=client,
            index_name="Document",
            text_key="text",
            embedding=embeddings,
            create_schema_if_missing=True
        )
        vectorstore.add_documents(docs)
    except Exception as e:
        raise

    # Step 4: Create Retriever
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        raise

    # Step 5: Initialize LLM (Groq - LLaMA3)
    try:
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )
    except Exception as e:
        raise

    # Step 6: Build RetrievalQA Chain
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        raise

    return qa
