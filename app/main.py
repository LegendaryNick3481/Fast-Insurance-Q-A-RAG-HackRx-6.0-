from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import gc
import asyncio

from app.utils import load_pdf
from app.qa_chain import get_qa_chain

from fastapi.concurrency import run_in_threadpool
import httpx

app = FastAPI()

TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


async def process_single_question(qa, question: str) -> str:
    """Process a single question with error handling"""
    try:
        answer = await run_in_threadpool(qa, question)
        return answer["result"]
    except Exception as e:
        return f"Error answering question: {str(e)}"


@app.post("/api/v1/hackrx/run")
async def run_query(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    temp_pdf = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client_http:
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

        # Process all questions in parallel
        tasks = [process_single_question(qa, question) for question in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that weren't caught in process_single_question
        final_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                final_answers.append(f"Error processing question {i + 1}: {str(answer)}")
            else:
                final_answers.append(answer)

        return {"answers": final_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    finally:
        if temp_pdf and os.path.exists(temp_pdf.name):
            os.remove(temp_pdf.name)
        gc.collect()


@app.get("/ping")
def ping():
    return {"status": "ok"}