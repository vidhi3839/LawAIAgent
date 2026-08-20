import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

load_dotenv()

from main import (
    run_query_with_retry,
    run_compound_query,
    run_multi_tool_query,
    detect_compound_question_node,
    save_thread_metadata,
    get_threads_for_lawyer,
    thread_belongs_to_lawyer,
    save_message,
    get_messages_for_thread,
    update_thread_summary,
    get_thread_summary,
)
from tasks.past_cases import ingest_single_case, list_uploaded_cases

app = FastAPI(title="Law Firm AI Agent API")

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    thread_id: str
    lawyer_name: str
    user_threshold: Optional[float] = None
    display_query: Optional[str] = None 


class QueryResponse(BaseModel):
    response: str
    confidence: float
    intent: str
    thread_id: str


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    try:
        result = None
        use_multi_tool = os.getenv("ENABLE_MULTI_TOOL", "false").lower() == "true"
        if use_multi_tool:
            thread_summary = get_thread_summary(request.thread_id)
            result = run_multi_tool_query(
                query=request.query,
                thread_id=request.thread_id,
                thread_summary=thread_summary,
                user_threshold=request.user_threshold,
            )
        if result is None:
            is_compound = detect_compound_question_node({"query": request.query}).get("is_possibly_compound")
            if is_compound:
                result = run_compound_query(
                    query=request.query,
                    thread_id=request.thread_id,
                    user_threshold=request.user_threshold,
                )
            if result is None:
                result = run_query_with_retry(
                    query=request.query,
                    thread_id=request.thread_id,
                    user_threshold=request.user_threshold,
                )
    except Exception as e:

        if "recursion" in str(e).lower() or "GraphRecursionError" in type(e).__name__:
            log.error(f"Graph hit its hard recursion_limit for thread_id={request.thread_id}: {e}")
            return QueryResponse(
                response="This request hit an internal safety limit before completing. Please try again.",
                confidence=0.0,
                intent="system_limit",
                thread_id=request.thread_id
            )
        raise

    final_response = result.get("best_response") or result["final_response"]
    confidence_score = result.get("best_confidence")
    if confidence_score is None:
        confidence_score = result["confidence_score"]

    display_intent = result.get("best_intent") or result.get("intent", "")
    if display_intent == "summarizer":
        doc_task = (result.get("document_result") or {}).get("confidence", {}).get("task")
        if doc_task == "qa":
            display_intent = "document_qa"

    save_thread_metadata(
        thread_id=request.thread_id,
        lawyer_name=request.lawyer_name,
        label=(request.display_query or request.query)[:40]
    )

    save_message(
        thread_id=request.thread_id,
        role="user",
        content=request.display_query or request.query
    )
    save_message(
        thread_id=request.thread_id,
        role="assistant",
        content=final_response,
        confidence=confidence_score,
        intent=display_intent
    )

    if confidence_score and confidence_score > 0.0:
        update_thread_summary(
            thread_id=request.thread_id,
            user_message=request.display_query or request.query,
            assistant_response=final_response,
        )

    return QueryResponse(
        response=final_response,
        confidence=confidence_score,
        intent=display_intent,
        thread_id=request.thread_id
    )


@app.post("/document")
async def upload_document(file: UploadFile = File(...), thread_id: str = "", lawyer_name: str = ""):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    save_thread_metadata(
        thread_id=thread_id,
        lawyer_name=lawyer_name,
        label=f"[Doc] {file.filename}"
    )

    return {
        "file_path": file_path,
        "file_name": file.filename,
        "thread_id": thread_id,
        "message": f"File uploaded successfully. Send '{file_path} summarise' or '{file_path} your question' to analyse."
    }


CASE_UPLOAD_DIR = "./data/case_pdfs_uploaded"
os.makedirs(CASE_UPLOAD_DIR, exist_ok=True)


@app.post("/past-cases/upload")
async def upload_past_case(
    file: UploadFile = File(...),
    case_name: str = Form(...),
    citation: str = Form(""),
    year: str = Form(""),
    court: str = Form(""),
    jurisdiction: str = Form(""),
    legal_issues: str = Form(""),
    lawyer_name: str = Form(""),
):

    """Lets a lawyer upload their own case PDF straight into the vector
    database used by the 'past cases' search feature. Metadata comes directly from the lawyer's form
    input here instead."""
    file_path = os.path.join(CASE_UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = ingest_single_case(
        file_path=file_path,
        case_name=case_name,
        citation=citation,
        year=year,
        court=court,
        jurisdiction=jurisdiction,
        legal_issues=legal_issues,
        uploaded_by=lawyer_name,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Ingestion failed for an unknown reason."))

    return {
        "case_key": result["case_key"],
        "chunks_ingested": result["chunks_ingested"],
        "auto_detected": result.get("auto_detected", {}),
        "message": f"'{case_name}' was successfully added to the case database ({result['chunks_ingested']} chunks).",
    }


@app.get("/past-cases/list")
def list_past_cases():
    """Returns every lawyer-uploaded case currently in the vector
    database, for display on the upload page."""
    return {"cases": list_uploaded_cases()}


@app.get("/threads")
def list_threads(lawyer_name: str):
    """Returns only the threads belonging to this lawyer, most recent first."""
    return {"threads": get_threads_for_lawyer(lawyer_name)}


@app.get("/threads/{thread_id}/messages")
def thread_messages(thread_id: str, lawyer_name: str):
    """Full transcript for one thread — used to repopulate the chat window
    when a lawyer reopens a past conversation. 404s if the thread doesn't
    belong to lawyer_name, so one lawyer can't pull another's history by
    guessing or reusing a thread_id."""
    if not thread_belongs_to_lawyer(thread_id, lawyer_name):
        raise HTTPException(status_code=404, detail="Thread not found for this lawyer.")
    return {"messages": get_messages_for_thread(thread_id)}


@app.get("/health")
def health():
    return {"status": "running"}

@app.get("/")
def root():
    return {"status": "Law Firm AI Agent API is running", "docs": "http://localhost:8000/docs"}