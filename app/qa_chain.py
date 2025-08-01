from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import Weaviate
from langchain_community.embeddings import VoyageAIEmbeddings
from weaviate.client import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
import os


def get_qa_chain(docs):
    print("🔧 Starting get_qa_chain()...", flush=True)

    # Step 1: Connect to Weaviate (v4)
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url:
        raise ValueError("❌ WEAVIATE_URL not set in environment.")

    try:
        print("🟡 Connecting to Weaviate (v4)...", flush=True)
        connection_params = ConnectionParams.from_params(
            http_host=weaviate_url.replace("https://", "").replace("http://", ""),
            http_secure=True
        )
        auth = AuthApiKey(weaviate_api_key) if weaviate_api_key else None
        client = WeaviateClient(
            connection_params=connection_params,
            auth=auth
        )
        print("✅ Weaviate connection established.", flush=True)
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}", flush=True)
        raise

    # Step 2: Create VoyageAI Embeddings
    try:
        print("🟡 Creating VoyageAI embeddings...", flush=True)
        embeddings = VoyageAIEmbeddings(
            model="voyage-3-large",
            voyage_api_key=os.getenv("VOYAGE_API_KEY")
        )
        print("✅ VoyageAI embeddings created.", flush=True)
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}", flush=True)
        raise

    # Step 3: Upload documents to Weaviate
    try:
        print(f"🟡 Uploading {len(docs)} documents to Weaviate...", flush=True)
        vectorstore = Weaviate(
            client=client,
            index_name="Document",  # Your Weaviate class name
            text_key="text",
            embedding=embeddings,
            create_schema_if_missing=True  # Optional: auto-create schema
        )
        vectorstore.add_documents(docs)
        print("✅ Documents stored in Weaviate.", flush=True)
    except Exception as e:
        print(f"❌ Error storing documents in Weaviate: {e}", flush=True)
        raise

    # Step 4: Create Retriever
    try:
        print("🟡 Creating retriever...", flush=True)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        print("✅ Retriever created.", flush=True)
    except Exception as e:
        print(f"❌ Error creating retriever: {e}", flush=True)
        raise

    # Step 5: Initialize LLM (Groq - LLaMA3)
    try:
        print("🟡 Initializing Groq LLM (LLaMA3)...", flush=True)
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )
        print("✅ Groq LLM initialized.", flush=True)
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}", flush=True)
        raise

    # Step 6: Build RetrievalQA Chain
    try:
        print("🟡 Building RetrievalQA chain...", flush=True)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ RetrievalQA chain created successfully.", flush=True)
    except Exception as e:
        print(f"❌ Error building RetrievalQA chain: {e}", flush=True)
        raise

    return qa
