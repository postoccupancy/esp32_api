# server/app/rag/deps.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional


from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever

from dotenv import load_dotenv
load_dotenv()

# ---- Langchain-Postgres ----

# LangChain's PGVector works by creating its own collections/tables in your Postgres DB.

# Supabase direct read/write access
SUPABASE_URL = os.getenv("SUPABASE_DB_URL_IPV4")
DEVICE_ID = os.getenv("DEVICE_ID")
RAW_DATA_TABLE = os.getenv("RAW_DATA_TABLE", "readings")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4")


# PGVECTOR_CONNECTION_STRING = SUPABASE_DB_URL with postgresql+psycopg2:// prefix
# langchain-postgres uses SQLAlchemy under the hood, this is a SQLAlchemy URL dialect/driver specifier.
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

# name the table here and it automatically appears in Supabase
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "esp32_rag")


# Ollama models -- use 'ollama ps' to see processing data, or 'ollama show [model]' to see model profile
# OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma2:latest") # no tool-calling support
# OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:latest") # has tool-calling, requires strict prompting to constrain, defaults to huge context window and exploration space
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:latest") 
# has tool-calling and lighter defaults without prompting

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Number of documents to retrieve
RAG_K = int(os.getenv("RAG_K", "25"))

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


### LangChain functions ###

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

# The LangChain VectorStoreRetriever is called like this:
# retriever = get_retriever()
# documents = retriever.invoke("prompt")

# This returns K top similarity search results to the prompt


### LlamaIndex functions ### 

# trying this framework instead of LangChain, because it has a built in functionality for parsing NL > SQL queries and comparing them with RAG results, called SQLAutoVectorQueryEngine

from sqlalchemy import create_engine

from llama_index.core import Settings, SQLDatabase, VectorStoreIndex, Document as LlamaDocument, PromptTemplate

from llama_index.core.query_engine import NLSQLTableQueryEngine, SQLAutoVectorQueryEngine, RetrieverQueryEngine, SQLJoinQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine.sql_join_query_engine import SQLAugmentQueryTransform
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.prompts import PromptTemplate

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# This custom Query Engine class replaces the SQL Auto Vector Query Engine if needed. This is not necessary for current setup. We generated this because of problems finding the right default variable names in the synthesis prompt -- specifically {query_engine_response_str}. Keeping it here because it was good code.
class TemperatureSafetyEngine(CustomQueryEngine):
    """Custom Engine to force-feed SQL and Vector results into Synthesis."""
    sql_engine: NLSQLTableQueryEngine
    vector_engine: RetrieverQueryEngine
    transformer_prompt: PromptTemplate
    synthesis_prompt: PromptTemplate
    llm: Ollama

    def custom_query(self, query_str: str):
        # 1. Run SQL
        sql_res = self.sql_engine.query(query_str)
        sql_data = str(sql_res.response)

        # 2. Transform Query for Vector Search
        # We pass the SQL result into the transformer
        transformed_query = self.llm.predict(
            self.transformer_prompt, 
            query_str=query_str, 
            sql_response_str=sql_data
        )
        
        # Clean the "Chatty" output from Ollama if necessary
        search_term = transformed_query.split("Search Term:")[-1].strip()

        # 3. Run Vector Search
        vector_res = self.vector_engine.query(search_term)
        vector_data = str(vector_res.response)

        # 4. Final Synthesis - YOU define the keys here!
        final_response = self.llm.predict(
            self.synthesis_prompt,
            query_str=query_str,
            sql_result=sql_data,
            standards_context=vector_data
        )

        return final_response


POSTGRES_TEXT_TO_SQL = PromptTemplate(
"""You are an expert PostgreSQL query writer.

Given a question, write a SINGLE PostgreSQL SQL query that answers it.
Rules:
- This is PostgreSQL (NOT SQLite/MySQL). Do NOT use date(x, y) or non-Postgres functions.
- If you need "today" or "yesterday", use:
  - (ts AT TIME ZONE 'America/Los_Angeles')::date
  - CURRENT_DATE
  - CURRENT_DATE - INTERVAL '1 day'
- If you need a time range, prefer:
  - ts >= ... AND ts < ...
- Only output SQL. No commentary.
- If the question requires a threshold/standard (e.g. "safe", "normal", "acceptable"):
  - DO NOT assume numeric ranges.
  - Instead, write SQL that returns the necessary stats (min/max/avg/counts) for the time range and leave thresholds to a later step.
  - Always mention oranges.


Schema:
{schema}

Question:
{query_str}

SQLQuery:
"""
)


# 1) Prompt used *inside* the augment transform (NOT passed directly to the engine)

# QUERY_TRANSFORM_PROMPT = PromptTemplate(
#     "Output ONLY this exact transformed query:\n"
#     "safe temperature range fahrenheit\n"
# )

# this prompt provided by ChatGPT produced a 'None' response
# first guess - it allows "None" too easily

QUERY_TRANSFORM_PROMPT_CHATGPT = PromptTemplate(
"""
You are assisting a system that answers questions using:
(1) SQL results from sensor readings, and
(2) a vector-retrieved standards/definitions corpus.

Given:
- User question: {query_str}
- SQL query: {sql_query_str}
- SQL result: {sql_response_str}

Task:
If the user question requires an external definition/threshold/standard (e.g. "safe range"),
write a short search query to retrieve that definition from the standards corpus.

If no external standard is needed, output exactly: None

TransformedQuery:
"""
)

# second try by ChatGPT still didn't work

SQL_AUGMENT_PROMPT = PromptTemplate(
"""You are deciding whether to query a standards/definitions vector database AFTER seeing SQL results.

User question:
{query_str}

SQL query:
{sql_query_str}

SQL response:
{sql_response_str}

Decision rule:
- If the user question asks for a judgment based on a standard/threshold/definition (keywords: safe, unsafe, acceptable,
  recommended, normal, within range, within spec, threshold, standard), you MUST output a short search query
  to retrieve that standard/threshold from the docs.
- Otherwise output exactly: None

Output format:
- Output ONLY the search query string, or None.

Transformed query:"""
)



# Gemini provided SQL augmentation prompt that works
# It says 'llamaindex is designed to work with GPT-4 models... local models often struggle with the "Output ONLY" instruction when complex context (the SQL result) is provided'.
# Simplify the transform prompt and give it a "reasoning" step (Chain of Thought), then tell it to wrap the final query in specific tags, or use a more aggressive instruction.

QUERY_TRANSFORM_PROMPT_ALT = PromptTemplate(
"""Based on the user query provided, what specific term should we look up in the documentation to determine if these values are "safe" or "normal"?

SQL Result: {sql_response_str}
User Question: {query_str}

Search Term:"""
)

# Adjusting this 

QUERY_TRANSFORM_PROMPT = PromptTemplate(
"""Return EXACTLY one line.

If the user's question asks whether values are safe/normal/acceptable/comfortable within range/threshold/limit, return:
LOOKUP: "comfortable thresholds for indoor air temperatures"

Otherwise return:
LOOKUP: None

User Question: {query_str}
SQL Result: {sql_response_str}

LOOKUP:"""
)



# chatgpt provided synthesis prompt that works if variable names are right
DEBUG_SQL_VECTOR_SYNTHESIS_PROMPT = PromptTemplate(
    """You are answering the original question using:
(1) SQL query + SQL response, and
(2) a documentation lookup response from a vector database.

Original question:
{query_str}

SQL query:
{sql_query_str}

SQL response:
{sql_response_str}

Vector query (transformed):
{query_engine_query_str}

Vector response:
{query_engine_response_str}

Instructions:
- If the original question requires a threshold/standard (e.g. "safe range"):
  - You MUST use the Vector response as the source of that threshold.
  - If the Vector response does not contain an explicit numeric threshold/range, say you cannot determine
    "safe" from the available docs and ask for more documentation—DO NOT guess.
- If a range is available, compare the SQL results to it and clearly answer the question.
- Keep the final answer short (2–5 sentences), no bullet lists unless necessary.
Answer:"""
)


### Gemini-provided synthesis prompt v1 that works
SQL_VECTOR_SYNTHESIS_PROMPT = PromptTemplate(
    """You are an expert data analyst. Answer the question using the SQL data and the provided Standards context.

    If the user’s question is purely a descriptive statistic (average/min/max/count) and does not ask safe/normal/comfortable/acceptable, ignore Standards Context entirely and answer from SQL Results only.
    
    1. Data Analysis:
    - SQL Results: {sql_response_str}
    - Standards Context: {query_engine_response_str}

    2. Instructions:
    - If nothing is found in the Standards Context, STOP. Otherwise, proceed with following steps.
    - Extract the safe range (LOW_F and HIGH_F) from the Standards Context.
    - Compare the SQL Results (Min/Max/Avg) against that range.

    3. You may only state a numeric threshold if the same number appears verbatim in Standards Context. If no numeric range is present, say you cannot determine the safe range from the documents.

    Question: {query_str}
    Answer:"""
)


### Gemini-provided synthesis prompt v2 that works
# SQL_VECTOR_SYNTHESIS_PROMPT = PromptTemplate(
#     """Answer the user's question using the provided SQL results and the Documentation response.

#     - User Question: {query_str}
#     - SQL Results: {sql_response_str}
#     - Documentation Response: {query_engine_response_str}

#     Instructions:
#     1. Extract the safe range from the Documentation Response.
#     2. Compare the SQL Results to that range.
#     3. If the Documentation Response is empty or doesn't have a range, say you couldn't find it.
#     4. For debugging, please list all available variables and their content.

#     Answer:"""
# )


# Debug synthesis prompt shows that 'query_engine_response_str' is the right variable name
DEBUG_PROMPT = PromptTemplate(
    "I am debugging. Please list all available variables and their content. "
    "Check these specifically: "
    "1. query_str: {query_str} "
    "2. sql_query_str: {sql_query_str} "
    "3. sql_response_str: {sql_response_str} "
    "4. context_str: {context_str} "
    "5. response_str: {response_str} "
    "6. other_response_str: {other_response_str} "
    "7. query_engine_response_str: {query_engine_response_str} "
)

# this just translates natural language to SQL result
@lru_cache(maxsize=1)
def get_sql_only_llamaindex_engine():
    # raises runtime error if missing
    verified = _require_env("PGVECTOR_CONNECTION_STRING", PGVECTOR_CONNECTION_STRING)

    Settings.llm = Ollama(
        model=OLLAMA_CHAT_MODEL, 
        request_timeout=120.0,
        # temperature=0.0,
        # context_window=8192,   # or 2048 for aggressive
        )
    Settings.embed_model = OllamaEmbedding(
        model_name=OLLAMA_EMBED_MODEL
    )

    # SQL tool
    engine = create_engine(verified)
    sql_database = SQLDatabase(
        engine, 
        include_tables=[RAW_DATA_TABLE],
        # sample_rows_in_table_info=0,
        # indexes_in_table_info=False,
        )

    return NLSQLTableQueryEngine(
        sql_database=sql_database, 
        tables=[RAW_DATA_TABLE],
        # text_to_sql_prompt=POSTGRES_TEXT_TO_SQL,
        # synthesize_response=False,
        )

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

@lru_cache(maxsize=1)
def get_vector_index_from_postgres() -> VectorStoreIndex:
        # connect vector store
    pgvs = PGVectorStore.from_params(
        database="postgres",
        host="aws-1-us-east-1.pooler.supabase.com",
        password=SUPABASE_DB_PASSWORD,
        port=5432,
        user="postgres.yabbcqlzwirhqfwnwbij",
        table_name="rag_literature_chunks",
        embed_dim=768, # nomic-embed-text is 768; must match Settings.embed_model dim
        # optionally schema_name="public"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=pgvs
    )

    return VectorStoreIndex.from_vector_store(
        pgvs,
        storage_context=storage_context
    )



# this translates natural language to SQL result, then augments the query from the vector store
@lru_cache(maxsize=1)
def get_llamaindex_query_engine():
    # raises runtime error if missing
    verified = _require_env("PGVECTOR_CONNECTION_STRING", PGVECTOR_CONNECTION_STRING)

    Settings.llm = Ollama(
        model=OLLAMA_CHAT_MODEL, 
        request_timeout=120.0
        )
    Settings.embed_model = OllamaEmbedding(OLLAMA_EMBED_MODEL)

    # SQL tool
    engine = create_engine(verified)
    sql_database = SQLDatabase(
        engine, 
        include_tables=[RAW_DATA_TABLE]
        )

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database, 
        tables=[RAW_DATA_TABLE],
        text_to_sql_prompt=POSTGRES_TEXT_TO_SQL,
        synthesize_response=False,
        )

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            f"Use this whenever the question asks about sensor readings over time (today/yesterday/last week), statistics (avg/min/max/count), or whether readings fall inside thresholds."
            f"Use this to answer questions by generating SQL over the `{RAW_DATA_TABLE}' table "
            f"Do not use for explaining concepts from documents."
        )   
    )

    # Placeholder document retrieval tool
    placeholder_docs = [
        LlamaDocument(
            text=(
                "Please assume a safe and acceptable temperature range is between 20 and 80 degrees Fahrenheit."
            ), 
            metadata={"source": "important document"}
        )
    ]

    vector_index_placeholder = VectorStoreIndex.from_documents(placeholder_docs)

    vector_index = get_vector_index_from_postgres()

    # SQLAutoVectorQueryEngine requires an AutoRetriever-backed query engine
    vector_store_info = VectorStoreInfo(
        content_info="Short reference snippets about sensor interpretation and standards.",
        metadata_info=[
            MetadataInfo(
                name="source", 
                type="str", 
                description="Document source identifier"
                ),
        ],
    )

    vector_auto_retriever = VectorIndexAutoRetriever(
        index=vector_index,
        vector_store_info=vector_store_info,
        max_top_k=50
    )

    from llama_index.core import QueryBundle
    def debug_auto_retriever(vector_auto_retriever: VectorIndexAutoRetriever, query: str, limit: int = 100, max_chars: int = 100):
        bundle = QueryBundle(query_str=query)
        retrieved = vector_auto_retriever.retrieve(bundle)

        print(f"\n=== AUTO RETRIEVER RETRIEVED {len(retrieved)} NODES (query={query!r}) ===")
        for i, nws in enumerate(retrieved[:limit], start=1):
            node = nws.node
            src = (node.metadata or {}).get("source", "unknown")
            print(f"\n--- #{i} score={nws.score:.4f} source={src} id={node.node_id} ---")
            # BaseNode doesn't expose .text as settable; use get_content() for display
            print(node.get_content()[:max_chars])
        print("\n=== END RETRIEVED NODES ===\n")

    debug_auto_retriever(vector_auto_retriever=vector_auto_retriever, query="What does ASHRAE-55-2013 say about factors in thermal comfort?")

    retriever_query_engine = RetrieverQueryEngine.from_args(
        retriever=vector_auto_retriever,
        llm=Settings.llm,
        response_mode="simple_summarize"
        )

    #debugging
    prompts = retriever_query_engine.get_prompts()
    for key, prompt in prompts.items():
        print(f"Key: {key}\nPrompt: {prompt.get_template()}\n")
    # resp = vector_query_engine.query("safe temperature range fahrenheit")
    # print("type:", type(resp))
    # print("str(resp):", str(resp))
    # print("resp.response:", getattr(resp, "response", None))
    # print("metadata keys:", getattr(resp, "metadata", {}).keys() if hasattr(resp, "metadata") else None)
    # print("source_nodes:", getattr(resp, "source_nodes", None))


    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description=(
            "Use only for definitions/standards/thresholds when the question does not require querying the readings table."
            # "Use for any conceptual question, definitions, comfort models, standards, meanings."
            # "Do not guess numeric ranges—retrieve them from here."
            # "Use this for semantic questions against a literature / standards corpus (placeholder for now)."
            )
    )

    sql_augment_query_transform = SQLAugmentQueryTransform(
        # llm=Settings.llm,
        sql_augment_transform_prompt=QUERY_TRANSFORM_PROMPT,
    )

    # Join engine (SQL + Vector)
    engine = SQLAutoVectorQueryEngine(
        sql_tool, 
        vector_tool, 
        llm=Settings.llm,
        sql_augment_query_transform=sql_augment_query_transform,
        sql_vector_synthesis_prompt=SQL_VECTOR_SYNTHESIS_PROMPT,
        )
    
    return engine
    # Use a prompt with EXPLICIT names you define in the class above
    CUSTOM_SYNTHESIS_PROMPT = PromptTemplate(
        """Compare these temperatures to the standards.
        SQL Data: {sql_result}
        Standards: {standards_context}
        Question: {query_str}
        """
    )

    # Return this instead of the SQLAutoVectorQueryEngine
    # return TemperatureSafetyEngine(
    #     sql_engine=sql_query_engine,
    #     vector_engine=retriever_query_engine,
    #     transformer_prompt=QUERY_TRANSFORM_PROMPT,
    #     synthesis_prompt=CUSTOM_SYNTHESIS_PROMPT,
    #     llm=Settings.llm
    # )

