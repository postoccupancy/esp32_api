from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.deps import get_llm, get_sql_only_llamaindex_engine, get_retriever, get_llamaindex_query_engine

import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import dateparser

TZ = ZoneInfo("America/Los_Angeles")

PARTS = {
    "morning": (6,12),
    "afternoon": (12,18),
    "evening": (18, 22),
    "night": (22, 6)
}

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

PLANNER_SYSTEM = """
You are a query planner for an IoT sensor dataset.
Given a user question, output ONLY valid JSON (no markdown) describing:
- intent: one of ["aggregate","retrieve","summarize"]
- metric: one of ["temperature_avg","temperature_min","temperature_max","humidity_avg","coverage_gaps","latest_snapshot","none"]
- time: an object describing the time range in natural terms.
Rules:
- If question references today/yesterday/last N hours/days, include that.
- If question references parts of day, use: ["morning","afternoon","evening","night"].
- If no time is given, default to last 24 hours.
- timezone should be "America/Los_Angeles".
JSON schema:
{
  "intent": "...",
  "metric": "...",
  "time": {
    "mode": "relative" | "absolute",
    "relative": {"unit":"hours|days","value":N} OR {"day":"today|yesterday","part_of_day":"morning|afternoon|evening|night"} ,
    "absolute": {"start":"YYYY-MM-DDTHH:MM:SS","end":"YYYY-MM-DDTHH:MM:SS"}
  },
  "timezone": "America/Los_Angeles"
}
"""

def plan_query(question: str) -> Dict[str, Any]:
    llm = get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": question}
        ]
    )
    text = response.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # fallback: treat as plain RAG summarize last 24h
        return {
            "intent": "summarize",
            "metric": "none",
            "time": {
                "mode": "relative",
                "relative": {
                    "unit": "hours",
                    "value": 24,
                }
            },
            "timezone": "America/Los_Angeles",
            "parse_error": str(e),
            "raw": text,
        }

def resolve_time_range(plan: Dict[str, Any]) -> tuple[datetime, datetime]:
    now_local = datetime.now(TZ)
    time_obj = plan.get("time", {})
    mode = time_obj.get("mode", "relative")

    # 1) Use structured plan if possible
    if mode == "absolute":
        start = datetime.fromisoformat(time_obj["absolute"]["start"]).replace(tzinfo=TZ)
        end = datetime.fromisoformat(time_obj["absolute"]["end"]).replace(tzinfo=TZ)
        return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

    relative = time_obj.get("relative", {})

    if "unit" in relative and "value" in relative:
        unit = relative["unit"]
        value = int(relative["value"])
        delta = timedelta(hours=value) if unit == "hours" else timedelta(days=value)
        start_local = now_local - delta
        end_local = now_local
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

    if "day" in relative:
        day = relative["day"]
        part = relative.get("part_of_day")
        base = now_local.date() if day == "today" else (now_local.date() - timedelta(days=1))
        if part:
            h0, h1 = PARTS[part]
            start_local = datetime(base.year, base.month, base.day, h0, 0, 0, tzinfo=TZ)
            end_local = datetime(base.year, base.month, base.day, h1 % 24, 0, 0, tzinfo=TZ)
            # if h1 == 24, bump day
            if h1 == 24:
                end_local = end_local + timedelta(days=1)
        else:
            start_local = datetime(base.year, base.month, base.day, 0, 0, 0, tzinfo=TZ)
            end_local = start_local + timedelta(days=1)
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


    # 2) Fallback: use dateparser on the raw question
    # Try to parse a datetime implied by phrases like "yesterday afternoon"
    dt = dateparser.parse(
        plan,
        settings={
            "RELATIVE_BASE": now_local,
            "TIMEZONE": "America/Los_Angeles",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "past",
        },
    )

    if dt:
        # If they didn't specify an explicit range, pick a reasonable window around it.
        # For “yesterday afternoon”, dateparser typically returns a dt in that period.
        start_local = dt.replace(minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(hours=6)
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


    # 3) Default
    start_local = now_local - timedelta(hours=24)
    end_local = now_local
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


import psycopg
from app.rag.rag_snapshots import SUPABASE_DB_URL, DEVICE_ID  # or read env again

def fetch_snapshots_between(
    start_utc: datetime,
    end_utc: datetime,
    device_id: str = DEVICE_ID,
    table: str = "snapshots",
) -> List[Document]:
    sql = f"""
      select window_start, window_end, snapshot_text
      from {table}
      where device_id = %s
        and window_start >= %s
        and window_end <= %s
      order by window_start asc;
    """
    docs: List[Document] = []
    with psycopg.connect(SUPABASE_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (device_id, start_utc, end_utc))
            for window_start, window_end, snapshot_text in cur.fetchall():
                docs.append(
                    Document(
                        page_content=snapshot_text,
                        metadata={
                            "kind": "hourly_snapshot",
                            "device_id": device_id,
                            "window_start": window_start.isoformat(),
                            "window_end": window_end.isoformat(),
                            "source": table,
                        },
                    )
                )
    return docs


AGG_METRICS = {
    "temperature_avg",
    "temperature_min",
    "temperature_max",
    "humidity_avg",
    "coverage_gaps",
    "latest_snapshot",
}

def llamaindex_answer_question(question: str) -> dict:
    # uses LlamaIndex 
    query_engine = get_llamaindex_query_engine()
    # query_engine = get_sql_only_llamaindex_engine()
    response = query_engine.query(question)
    return {
        "question": question,
        "response": str(response),
        "answer": getattr(response, "response", str(response)),
        "metadata": getattr(response, "metadata", None)
    }



def _langchain_answer_question(question: str) -> Dict[str, Any]:
    llm = get_llm()
    plan = plan_query(question)
    metric = plan.get("metric", "none")

    # If it’s an aggregate/time-bound question, fetch the correct docs deterministically
    if metric in AGG_METRICS:
        start_utc, end_utc = resolve_time_range(plan)
        documents = fetch_snapshots_between(start_utc, end_utc)

        if not documents:
            return {
                "question": question,
                "answer": f"No snapshots found between {start_utc.isoformat()} and {end_utc.isoformat()}.",
                "sources": [],
                "retrieved": 0,
                "plan": plan,
                "time_range_utc": {"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            }        

        # Deterministic answers for common aggregates (recommended)
        if metric == "latest_snapshot":
            md = documents[-1].metadata
            answer = f"Most recent snapshot in range: {md.get('window_start')} - {md.get('window_end')}."
            return {
                "question": question,
                "answer": answer,
                "sources": _list_snapshots_used([documents[-1]]),
                "retrieved": len(documents),
                "plan": plan,
                "time_range_utc": {"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            }

        # Build LLM prompt to summarize based on the correct docs:
        prompt = _build_prompt(question, documents)
        response = llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "sources": _list_snapshots_used(documents),
            "retrieved": len(documents),
            "plan": plan,
            "time_range_utc": {"start": start_utc.isoformat(), "end": end_utc.isoformat()},
        }
    
    # Otherwise: semantic RAG top-k search
    retriever = get_retriever()
    documents = retriever.invoke(question)
    prompt = _build_prompt(question, documents)
    response = llm.invoke(prompt)
    print(response.content)

    return {
        "question": question,
        "answer": response.content,
        "sources": _list_snapshots_used(documents),
        "retrieved": len(documents),
    }

from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool, ToolMetadata

# Standard LlamaIndex Chat Engines (like SimpleChatEngine, ContextChatEngine) are often too rigid for the SQL+Vector combo because they expect a simple retriever.
# An Agent on the other hand treats the engine as a "tool" it can call.
# ReACT agent is great for CLI use because it reasons through steps

def build_agent():
    query_engine = get_llamaindex_query_engine()

    def temperature_analyst(question: str) -> str:
        response: Response = query_engine.query(question)
        print_sources(response)
        return str(response)

    tool = FunctionTool.from_defaults(
        fn=temperature_analyst,
        name="temperature_analyst",
        description="""Analyzes temperature readings from the database and compares them to reference standards in the documents.
        Input is a natural language question ONLY. Do not pass JSON arguments.
        """,

    )

    agent = ReActAgent(
        tools=[tool],
        llm=Settings.llm,
        verbose=True,
    )

    context = Context(agent)
    
    return agent, context


import os
from llama_index.core.base.response.schema import Response
RAG_K = int(os.getenv("RAG_K", "25"))

def print_sources(resp: Response, max_chars: int = 800):
    print("\n=== SOURCE NODES USED ===")
    for i, nws in enumerate(resp.source_nodes, 0):
        node = nws.node
        src = node.metadata.get("source", "unknown")
        print(f"\n--- #{i} score={nws.score:.4f} source={src} ---")
        print(node.get_content()[:max_chars])
    print("\n=== END SOURCES ===\n")

# In the Workflow-based Agent, the .chat() method is typically asynchronous because Workflows are designed to handle long-running events and streaming.

import asyncio
async def start_chat():
    agent, context = build_agent()
    print("Type 'exit' to quit.\n")

    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break
        handler = agent.run(msg, ctx=context)

        # Stream tokens (optional)
        # optional UI polish - emit events (generated text) as they occur
        async for event in handler.stream_events():
            if isinstance(event, AgentStream): # only print 'generated text deltas' not tool call start/end, name, arguments, steps, errors, retries, etc.
                print(event.delta, end="", flush=True)

        response = await handler
        print(f"\n\n")


def just_answer():
    question = "What was the average temperature yesterday?"
    result = llamaindex_answer_question(question)
    print("question: ", result["question"])
    print("response: ", result["response"])
    print("metadata: ", result["metadata"])




if __name__ == "__main__":
    asyncio.run(start_chat())
    # just_answer()
