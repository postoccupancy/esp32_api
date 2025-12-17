import os
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import DictCursor
from openai import OpenAI
import requests
from rag_snapshots import get_readings_since, build_snapshot

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4")

### using OpenAI -- no free tier ###
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)
# def embed_text(text: str):
#     # Using OpenAI embedding, returns a list[float] of length 1536
#     resp = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text,
#     )
#     return resp.data[0].embedding

### using Ollama local server ###
def embed_text(text):
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return resp.json()["embedding"]


def upsert_snapshot(device_id: str, hours: int = 6):
    # 1) Build snapshot
    rows = get_readings_since(device_id, hours=hours)
    if not rows:
        print(f"No rows for {device_id} in last {hours} hours.")
        return

    snapshot_text, window_start, window_end = build_snapshot(device_id, rows)

    # 2) Generate embedding
    embedding = embed_text(snapshot_text)

    # 3) Insert into snapshots
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor()

    print("embedding type:", type(embedding))
    print("embedding length:", len(embedding))
    print("first few values:", embedding[:5])

    insert_sql = """
        insert into snapshots (
            device_id,
            window_start,
            window_end,
            snapshot_text,
            embedding
        )
        values (%s, %s, %s, %s, %s::vector)
        returning id;
    """

    cur.execute(
        insert_sql,
        (
            device_id,
            window_start,
            window_end,
            snapshot_text,
            embedding,
        ),
    )

    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted snapshot id={new_id} for {device_id}")
    return new_id


if __name__ == "__main__":
    # Example: index last 6h for one device
    upsert_snapshot("esp32-s3-devkit-001", hours=6)