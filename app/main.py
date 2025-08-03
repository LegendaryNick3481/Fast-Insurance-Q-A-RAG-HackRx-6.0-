from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import gc

from app.utils import load_pdf
from app.qa_chain import get_qa_chain

from fastapi.concurrency import run_in_threadpool
import httpx

app = FastAPI()

TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    temp_pdf = None
    try:
        async with httpx.AsyncClient() as client_http:
            pdf_response = await client_http.get(request.documents)
            pdf_response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_response.content)
                pdf_path = temp_pdf.name

        # Load and split PDF into documents
        docs = await load_pdf(pdf_path)

        if not docs:
            raise ValueError("No content extracted from PDF")

        # Get QA chain using hybrid + rerank retriever
        qa = await get_qa_chain(docs)

        # Run QA loop
        answers = []
        for q in request.questions:
            try:
                answer = await run_in_threadpool(qa, q)
                answers.append(answer["result"])
            except Exception as e:
                answers.append(f"Error answering question: {str(e)}")

        return {"answers": answers}

    finally:
        if temp_pdf and os.path.exists(temp_pdf.name):
            os.remove(temp_pdf.name)
        gc.collect()

@app.get("/ping")
def ping():
    return {"status": "ok"}
