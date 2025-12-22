from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.deps import get_llm, get_retriever

# adding here for local testing, remove when deploying
import os
from dotenv import load_dotenv
load_dotenv()
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
DEVICE_ID = os.getenv("DEVICE_ID")

@dataclass
class Source:
    kind: str
    device_id: str
    window_start: str
    window_end: str
    source: str

def _list_snapshots_used(documents: List[Document]) -> List[Source]:
    sources: List[Source] = []
    for document in documents:
        md = document.metadata or {} # if empty, md.get() will return None
        sources.append(
            Source(
                kind=md.get("kind"),
                device_id=md.get("device_id"),
                window_start=md.get("window_start"),
                window_end=md.get("window_end"),
                source=md.get("source"),
            )
        )
    return sources


def _build_prompt(question: str, documents: List[Document]) -> str:

    context_blocks = []

    for index, document in enumerate(documents, start=1):
        md = document.metadata or {}

        cite = f"[{index}] device={md.get('device_id')} {md.get('window_start')}→{md.get('window_end')}"

        context_blocks.append(f"{cite}\n{document.page_content}".strip())

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return (
        "You are an assistant analyzing hourly sensor snapshots.\n"
        "Use ONLY the provided context. If the answer is not supported, say so.\n"
        "When you make a claim, cite sources like [1], [2].\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:\n"
    )


def answer_question(question: str) -> Dict[str, Any]:
    retriever = get_retriever()
    llm = get_llm()

    documents = retriever.invoke(question)

    prompt = _build_prompt(question, documents)
    resp = llm.invoke(prompt)

    print(resp.content)
    return {
        "question": question,
        "answer": resp.content,
        "sources": _list_snapshots_used(documents),
        "retrieved": len(documents),
    }

if __name__ == "__main__":
    question = "What was the average temperature yesterday afternoon?"
    result = answer_question(question)
    print(result)
