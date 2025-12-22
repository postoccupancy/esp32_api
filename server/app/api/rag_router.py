import os
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime, timezone

from app.rag.rag_query import answer_question
from app.rag.rag_snapshots import build_langchain_documents
from app.rag.rag_index import archive_documents_in_database, index_documents_in_vectorstore

from fastapi import Header, HTTPException


### FastAPI authentication layer ###
# 1. FastAPI maps custom header like X-RAG-Token to x_rag_token parameter
# 2. Attach dependencies=[Depends(...)] to the route to run a check before endpoint handler
# 3. server checks incoming requests for a specific header, 
# e.g. X-RAG-Token: a-long-secret-key
# 4. if it doesn't match, return 401 Unauthorized

RAG_CRON_TOKEN = os.getenv("RAG_CRON_TOKEN", "a-long-secret-key")

# FastAPI will look for X-RAG-Token based on 'x_rag_token' parameter
def require_cron_token(x_rag_token: str = Header(default="")):
    if not RAG_CRON_TOKEN or x_rag_token != RAG_CRON_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

### To call these endpoints from terminal, use:
# curl -X POST http://127.0.0.1:8000/rag/[endpoint] -H "X-RAG-Token: [token]"


router = APIRouter()

class QueryReq(BaseModel):
    question: str

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4")
DEVICE_ID = os.getenv("DEVICE_ID")

@router.post("/query")
def rag_query(request: QueryReq):
    return answer_question(request.question)

@router.post("/rebuild", dependencies=[Depends(require_cron_token)])
def rag_rebuild():
    documents = build_langchain_documents()
    if not documents:
        return {"ok": True, "snapshots_indexed": 0, "note": "No data to index from all time."}
    index_documents_in_vectorstore(documents)
    archive_documents_in_database(SUPABASE_DB_URL, documents)
    return {"ok": True, "snapshots_indexed": len(documents)}

@router.post("/index", dependencies=[Depends(require_cron_token)])
def rag_index():
    documents = build_langchain_documents(lookback_hours=1)
    if not documents:
        return {"ok": True, "snapshots_indexed": 0, "note": "No data to index in the last hour."}
    index_documents_in_vectorstore(documents)
    archive_documents_in_database(SUPABASE_DB_URL, documents)
    return {"ok": True, "snapshots_indexed": len(documents)}

