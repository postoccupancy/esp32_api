# rag_query.py
import os
from typing import List

import psycopg2
from psycopg2.extras import DictCursor
import requests
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file

from rag_index import upsert_snapshot  # <-- new import

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4")  # same pooler URL you use in rag_index.py

# --- Embeddings via Ollama (same model as rag_index.py) ---

def embed_text(text: str) -> List[float]:
    """Get an embedding from Ollama's nomic-embed-text model."""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "embedding" not in data:
        raise RuntimeError(f"No 'embedding' key in Ollama response: {data}")

    embedding = data["embedding"]

    if not isinstance(embedding, list) or len(embedding) == 0:
        raise RuntimeError(f"Got empty or invalid embedding: {embedding}")

    return embedding


def embedding_to_vector_literal(embedding: List[float]) -> str:
    """
    Convert a Python list of floats into a pgvector-compatible literal, e.g.:
    [0.1, 0.2] -> "[0.1,0.2]"
    """
    if not embedding:
        raise ValueError("Cannot convert empty embedding to vector literal")
    return "[" + ",".join(str(x) for x in embedding) + "]"

# --- Ensure we have a recent snapshot for this device ---

def ensure_recent_snapshot(
    device_id: str,
    max_age_minutes: int = 60,
    hours_window: int = 6,
) -> None:
    """
    Make sure there's at least one snapshot for this device whose window_end
    is not older than `max_age_minutes`. If none exist or it's too old,
    call upsert_snapshot(...) to create a fresh one.
    """
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor(cursor_factory=DictCursor)

    sql = """
        select window_end
        from snapshots
        where device_id = %s
        order by window_end desc
        limit 1;
    """
    cur.execute(sql, (device_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    now = datetime.now(timezone.utc)

    if row is None:
        print(f"No snapshots found for {device_id}, creating one now...")
        upsert_snapshot(device_id, hours=hours_window)
        return

    last_end = row["window_end"]
    # Make sure last_end is timezone-aware
    if last_end.tzinfo is None:
        last_end = last_end.replace(tzinfo=timezone.utc)

    age_minutes = (now - last_end).total_seconds() / 60.0

    if age_minutes > max_age_minutes:
        print(
            f"Last snapshot for {device_id} is {age_minutes:.1f} min old "
            f"(limit {max_age_minutes} min); refreshing..."
        )
        upsert_snapshot(device_id, hours=hours_window)
    else:
        print(
            f"Snapshot for {device_id} is fresh ({age_minutes:.1f} min old); "
            "no refresh needed."
        )

# --- Retrieval from pgvector ---

def retrieve_snapshots(question: str, device_id: str, k: int = 5):
    """
    Embed the question, then retrieve the top-k most similar snapshots
    for the given device_id from the snapshots table.
    """
    query_embedding = embed_text(question)
    query_literal = embedding_to_vector_literal(query_embedding)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor(cursor_factory=DictCursor)

    sql = """
        select
          id,
          device_id,
          window_start,
          window_end,
          snapshot_text,
          1 - (embedding <=> %s::vector) as similarity
        from snapshots
        where device_id = %s
        order by embedding <=> %s::vector
        limit %s;
    """

    cur.execute(sql, (query_literal, device_id, query_literal, k))
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows


# --- Answer generation via Ollama chat model ---

def answer_with_llama(question: str, snapshots) -> str:
    """
    Use an Ollama chat model (e.g. llama3) to answer the question
    given a set of snapshot rows.
    """
    # Build context from snapshots
    if not snapshots:
        context = "No sensor snapshots were retrieved from the database."
    else:
        blocks = []
        for s in snapshots:
            block = f"""[Snapshot id={s['id']}]
Device: {s['device_id']}
Window: {s['window_start']} – {s['window_end']}

{s['snapshot_text']}
"""
            blocks.append(block.strip())
        context = "\n\n---\n\n".join(blocks)

    system_prompt = (
        "You are an assistant that analyzes environmental sensor data "
        "(temperature and humidity) and explains patterns clearly for a "
        "non-technical user. Be honest about uncertainty."
    )

    user_prompt = f"""
User question:
{question}

Context from sensor snapshots:
{context}
""".strip()

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3",  # make sure you've pulled this model: `ollama pull llama3`
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat response format: {"message":{"role":"assistant","content":"..."},"done":true}
    msg = data.get("message", {})
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"No content in Ollama chat response: {data}")

    return content


def answer_question(
        question: str, 
        device_id: str = "esp32-s3-devkit-001", 
        k: int = 5,
        max_age_minutes: int = 60,
        hours_window: int = 6,
        ) -> str:
    """
    High-level RAG pipeline:
    0) Ensure we have at least one fresh snapshot for this device
    1) Retrieve the most relevant snapshots for this device
    2) Ask an LLM to answer using those snapshots as context
    """
    ensure_recent_snapshot(device_id, max_age_minutes=max_age_minutes, hours_window=hours_window)
    snapshots = retrieve_snapshots(question, device_id=device_id, k=k)
    return answer_with_llama(question, snapshots)


if __name__ == "__main__":
    q1 = "Is there anything unusual to report today compared with all time?"
    q = "What is a typical range of temperatures and humidity for homes in the pacific northwest? does it vary by region?"
    answer = answer_question(q, device_id="esp32-s3-devkit-001", k=3)
    print("\nQUESTION:\n", q)
    print("\nANSWER:\n", answer)
