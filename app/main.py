from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
import os

from app.utils import load_pdf  # Make sure this function returns LangChain documents

from langchain.prompts import PromptTemplate
from langchain_weaviate import WeaviateVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc

app = FastAPI()

# 🔐 Team token for auth
TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"

# 🌱 Load env vars
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# ✅ Initialize Weaviate client (v4 syntax)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)


# 📦 Request body model
class QueryRequest(BaseModel):
    documents: str  # URL of PDF
    questions: List[str]


@app.post("/api/v1/hackrx/run")
def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    print("🔒 Received request at /api/v1/hackrx/run", flush=True)

    # 🔐 Authorization check
    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        print("❌ Unauthorized access attempt", flush=True)
        raise HTTPException(status_code=401, detail="Unauthorized")
    print("✅ Authorization successful", flush=True)

    # 📥 Download PDF file
    try:
        print(f"📥 Downloading PDF from: {request.documents}", flush=True)
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_response.content)
            pdf_path = temp_pdf.name
        print(f"✅ PDF downloaded to temporary path: {pdf_path}", flush=True)
    except Exception as e:
        print(f"❌ PDF download failed: {e}", flush=True)
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")

    try:
        # 📄 Extract content from PDF
        print("📄 Extracting documents from PDF...", flush=True)
        docs = load_pdf(pdf_path)
        print(f"✅ Extracted {len(docs)} documents from PDF", flush=True)

        if not docs:
            raise ValueError("No content extracted from PDF")

        # 🧠 Setup embedding model
        embeddings = VoyageAIEmbeddings(model="voyage-3-large", voyage_api_key=VOYAGE_API_KEY)

        # 🔗 Create vector store with Weaviate v4
        print("🧠 Creating Weaviate vector store...", flush=True)
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="Document",
            text_key="text",
            embedding=embeddings,
        )

        vectorstore.add_documents(docs)
        print("✅ Documents added to Weaviate", flush=True)

        # 🤖 Setup RetrievalQA with Groq LLM - MAP_REDUCE VERSION
        print("🤖 Initializing QA chain with Groq LLM...", flush=True)

        # Map phase prompt - processes each document chunk individually
        map_template = """You are analyzing an insurance policy document. Based on the following context, extract any information relevant to the question. If no relevant information is found, respond with "No relevant information found."

Context:
{context}

Question: {question}

Relevant information:"""

        map_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=map_template
        )

        # Reduce phase prompt - combines results from all chunks
        reduce_template = """You are a helpful assistant answering questions based strictly on the provided insurance policy document.

Based on the following extracted information from different parts of the document, provide a direct and factual answer to the question. Do not say things like "according to the context." If the answer is not found, reply: "Not mentioned in the policy document."

Use no more than 2 sentences.

Extracted information:
{summaries}

Question: {question}

Answer:"""

        reduce_prompt = PromptTemplate(
            input_variables=["summaries", "question"],
            template=reduce_template
        )

        qa = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                api_key=GROQ_API_KEY
            ),
            retriever=vectorstore.as_retriever(),
            chain_type="map_reduce",
            chain_type_kwargs={
                "question_prompt": map_prompt,
                "combine_prompt": reduce_prompt
            },
            return_source_documents=False
        )

        print("✅ QA chain ready", flush=True)

        # ❓ Run questions
        print("❓ Processing questions...", flush=True)
        answers = []
        for idx, q in enumerate(request.questions, start=1):
            print(f"🔍 Q{idx}: {q}", flush=True)
            try:
                result = qa(q)
                answers.append(result["result"])
                print(f"✅ A{idx}: {result['result']}", flush=True)
            except Exception as e:
                print(f"❌ Error answering question {idx}: {e}", flush=True)
                answers.append(f"Error answering question: {str(e)}")

        return {"answers": answers}

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"🧹 Deleted temporary file: {pdf_path}", flush=True)


# Graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    client.close()