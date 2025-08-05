from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import gc
import asyncio
import time

from app.utils import load_pdf_ultra_fast, cleanup_temp_files
from app.qa_chain import get_ultra_fast_qa_chain, cleanup_client

from fastapi.concurrency import run_in_threadpool
import httpx

from app.rerankers.local_cross_encoder import initialize_cross_encoder


app = FastAPI()

TEAM_TOKEN = "8ad62148045cbf8137a66e1d8c0974e14f62a970b4fa91afb850f461abfbadb8"

SEM_LIMIT = 10
semaphore = asyncio.Semaphore(SEM_LIMIT)


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


async def process_single_question_ultra_fast(qa, question: str, timeout: int = 8) -> str:
    async with semaphore:
        try:
            answer = await run_in_threadpool(qa.invoke, {"query": question})
            return answer["result"]
        except asyncio.TimeoutError:
            return f"Timeout: Question processing exceeded {timeout}s"
        except Exception as e:
            return f"Error: {str(e)}"


@app.post("/api/v1/hackrx/run")
async def run_query_ultra_fast(
    request: QueryRequest,
    authorization: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    start_time = time.time()

    if not authorization or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    temp_pdf_path = None
    cleanup_fn = None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)
        ) as client_http:
            pdf_response = await client_http.get(request.documents, follow_redirects=True)
            pdf_response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_response.content)
                temp_pdf_path = temp_pdf.name

        docs = await load_pdf_ultra_fast(temp_pdf_path)
        if not docs:
            raise ValueError("No content extracted from PDF")

        qa, cleanup_fn = await get_ultra_fast_qa_chain(docs)

        max_concurrent = min(len(request.questions), 12)
        per_question_timeout = max(6, min(10, 80 // len(request.questions)))

        if len(request.questions) <= max_concurrent:
            tasks = [
                process_single_question_ultra_fast(qa, q, per_question_timeout)
                for q in request.questions
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for i in range(0, len(request.questions), max_concurrent):
                batch = request.questions[i:i + max_concurrent]
                tasks = [
                    process_single_question_ultra_fast(qa, q, per_question_timeout)
                    for q in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)

        answers = [
            f"Error: {str(r)}" if isinstance(r, Exception) else str(r)
            for r in results
        ]

        total_time = time.time() - start_time

        if cleanup_fn:
            background_tasks.add_task(cleanup_fn)

        # Group all questions and answers into separate dicts
        questions_dict = {f"q{i + 1}": q for i, q in enumerate(request.questions)}
        answers_dict = {f"a{i + 1}": a for i, a in enumerate(answers)}

        print("Questions:", questions_dict)
        print("Answers:", answers_dict)

        return {
            "answers": answers
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Ultra-fast timeout exceeded")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if temp_pdf_path:
            cleanup_temp_files(temp_pdf_path)
        gc.collect()


@app.get("/ping")
def ping():
    return {"status": "ultra-fast", "mode": "lightning"}


@app.get("/health")
async def health_check():
    return {
        "status": "ultra-optimized",
        "target": "sub-15s processing",
        "optimizations": [
            "llama-3.1-8b-instant",
            "massive chunk sizes",
            "128 batch embedding upload",
            "max 50 chunks per document",
            "12 concurrent questions",
            "background cleanup"
        ]
    }

@app.on_event("startup")
async def startup_event():
    await run_in_threadpool(initialize_cross_encoder)

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_client()