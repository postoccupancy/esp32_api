from __future__ import annotations

from typing import List
from datetime import datetime
import psycopg
from langchain_core.documents import Document

from app.rag.deps import get_vectorstore

test_document = Document(
    page_content="",
    metadata={
        "device_id": "device_id",
        "window_start": "window_start",
        "window_end": "window_end",
    },
)



def _stable_doc_id(document: Document) -> str:
    # Stable ID prevents duplicates when you rebuild.
    # .metadata["name"] is for required fields, fails loudly if missing.
    # .metadata.get("name") is for optional fields, returns None if missing.
    device_id = document.metadata["device_id"]
    window_start = str(document.metadata["window_start"])  # ISO string

    # if not device_id or not window_start:
    #     raise ValueError(
    #         f"Document missing required metadata keys. "
    #         f"Expected device_id and window_start, got: {document.metadata}"
    #     )

    return f"{device_id}:{window_start}:hourly"

_stable_doc_id(test_document)

def archive_documents_in_database(
        db_url: str, 
        documents: List[Document],
        archive: str = "snapshots",

        ) -> int:
    """
    Keeps a human-friendly archive of snapshot text in your own table.
    This is separate from the LangChain PGVector tables.
    """

    # the 'on conflict ... do update' lines prevent duplicates
    # it requires there be a unique constraint in the table setup on (device_id, window_start, window_end)
    sql = f"""
      insert into {archive} (
        device_id, 
        window_start, 
        window_end, 
        snapshot_text
        )
      values (%s, %s, %s, %s)
      on conflict (device_id, window_start, window_end)
      do update set snapshot_text = excluded.snapshot_text
    """

    connection = psycopg.connect(db_url)
    cursor = connection.cursor()

    try:
        for document in documents:
            metadata = document.metadata or {}
            cursor.execute(
                sql,
                (
                    metadata["device_id"],
                    metadata["window_start"],
                    metadata["window_end"],
                    document.page_content,
                ),
            )
        connection.commit()
    finally:
        cursor.close()
        connection.close()

    return len(documents)


def index_documents_in_vectorstore(documents: List[Document]) -> int:
    """
    Index into LangChain's PGVector-backed vectorstore.
    This stores embeddings + text + metadata in LangChain-managed tables.
    """
    vectorstore = get_vectorstore() # returns a PGVector object

    # stable id's ensures no duplicates
    document_ids = [_stable_doc_id(document) for document in documents]

    # LangChain will: embed -> write to Postgres tables -> associate IDs
    vectorstore.add_documents(documents, ids=document_ids)

    return len(documents)
