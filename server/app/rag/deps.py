# server/app/rag/deps.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever

# adding here for local testing, remove when deploying and call once in main.py
import os
from dotenv import load_dotenv
load_dotenv()



# ---- Langchain-Postgres ----

# LangChain's PGVector works by creating its own collections/tables in your Postgres DB.

# Supabase direct read/write access
SUPABASE_URL = os.getenv("SUPABASE_DB_URL_IPV4")
DEVICE_ID = os.getenv("DEVICE_ID")

# PGVECTOR_CONNECTION_STRING = SUPABASE_DB_URL with postgresql+psycopg2:// prefix
# langchain-postgres uses SQLAlchemy under the hood, this is a SQLAlchemy URL dialect/driver specifier.
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

# name the table here and it automatically appears in Supabase
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "esp32_rag")


# Ollama models
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Retrieval defaults
RAG_K = int(os.getenv("RAG_K", "3"))

# error handling for missing cnnection string
def _require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(
            f"Missing required env var: {name}. "
            f"Set {name} to your Supabase Postgres connection string "
            f"(e.g., postgresql+psycopg://...)."
        )
    return value


# ---- Singletons (cached) ----
# the lru_cache decorator ensures we only create one instance and subsequent calls return the same object.
# A singleton is "one shared instance"

@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:

    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL
        )


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:

    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        temperature=0.9,
    )

@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    # raises runtime error if missing
    verified = _require_env("PGVECTOR_CONNECTION_STRING", PGVECTOR_CONNECTION_STRING)

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=PGVECTOR_COLLECTION,
        connection=verified,
        # use_jsonb=True, # optional
    )


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    vs = get_vectorstore()

    # a retriever is a configured view of a vectorstore
    # you don't construct it independently; you ask for one from the vectorstore 
    return vs.as_retriever(
        search_kwargs={"k": RAG_K}
        )

# By having the vectorstore create the retriever, LangChain can translate the different query API, metadata handling, etc. of different vectorstore providers (e.g., PGVector, Chroma, FAISS, Pinecone) to a consistent interface.