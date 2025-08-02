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

TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_response.content)
            pdf_path = temp_pdf.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download failed: {e}")

    try:
        docs = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No content extracted from PDF")

        embeddings = VoyageAIEmbeddings(model="voyage-3-large", voyage_api_key=VOYAGE_API_KEY)

        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="Document",
            text_key="text",
            embedding=embeddings,
        )

        vectorstore.add_documents(docs)

        map_template = """You are analyzing an insurance policy document. Based on the following context, extract any information relevant to the question. If no relevant information is found, respond with "No relevant information found."

Context:
{context}

Question: {question}

Relevant information:"""

        map_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=map_template
        )

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

        answers = []
        for q in request.questions:
            try:
                result = qa(q)
                answers.append(result["result"])
            except Exception as e:
                answers.append(f"Error answering question: {str(e)}")

        return {"answers": answers}

    finally:

        import gc

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        for var in ['docs', 'embeddings', 'vectorstore', 'qa', 'answers']:

            if var in locals():
                del locals()[var]

gc.collect()


@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.on_event("shutdown")
def shutdown_event():
    client.close()
