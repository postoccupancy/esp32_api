## 1.) Choose the right document loaders

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Document

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "docs" / "pdfs"

## 2.) Fetch your bibliography into raw documents

pdf_docs = SimpleDirectoryReader(
    input_dir=str(PDF_DIR),
    recursive=True,
).load_data()

# Each element is a Document with:
# .text
# .metadata["file_name"]
# .metadata["file_path"]

urls = [
    "https://en.wikipedia.org/wiki/ASHRAE_55",
    "https://www.mdpi.com/2075-5309/11/8/336",
]

web_docs = SimpleWebPageReader().load_data(urls)


# Normalize metadata (important for retrieval)

def normalize_metadata(doc: Document) -> Document:
    text = doc.get_content()
    meta = dict(doc.metadata or {})

    source = meta.get("file_name") or meta.get("url", "unknown")

    if "ASHRAE" in source or "ASHRAE" in text:
        meta["category"] = "standard"
        meta["organization"] = "ASHRAE"
    elif "WMO" in text or "WMO" in source:
        meta["category"] = "guideline"
        meta["organization"] = "WMO"
    else:
        meta["category"] = "paper"

    meta["source"] = source
    return Document(text=text, metadata=meta)

raw_docs = [normalize_metadata(d) for d in pdf_docs + web_docs]

import re
# clean out NUL (0x00) characters and other PDF junk
def clean_text(s: str) -> str:
    if not s:
        return s
    # Remove NUL bytes (Postgres can't store them)
    s = s.replace("\x00", "")
    s = s.replace("\ufeff", "")  # BOM
    # collapse weird whitespace
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


## 3.) Chunk the documents

import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore  # pip install llama-index-vector-stores-postgres

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# PGVECTOR_CONNECTION_STRING = SUPABASE_DB_URL with postgresql+psycopg2:// prefix
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

# name the table here and it automatically appears in Supabase
LLAMAINDEX_PGVECTOR_COLLECTION = os.getenv("LLAMAINDEX_PGVECTOR_COLLECTION", "rag_literature_chunks")

SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = os.getenv("EMBED_DIM", 768)

from llama_index.core.schema import TextNode

def ingest():
    embed_model = OllamaEmbedding(
        model_name=OLLAMA_EMBED_MODEL
    )

    splitter = SentenceSplitter(
        chunk_size=256, # tokens -- don't over-chunk, 512-768 tokens is the sweet spot? But got error "the input length exceeds the context length"
        chunk_overlap=50, # use overlap so definitions aren't split away from context
    )

    # give each node a node.text and node.metadata (source, category, etc)
    nodes = splitter.get_nodes_from_documents(raw_docs)

    # do a cleaning step on PDF junk
    for n in nodes:
        if isinstance(n, TextNode):
            n.text = clean_text(n.text)

    # connect vector store
    pgvs = PGVectorStore.from_params(
        database="postgres",
        host="aws-1-us-east-1.pooler.supabase.com",
        password=SUPABASE_DB_PASSWORD,
        port=5432,
        user="postgres.yabbcqlzwirhqfwnwbij",
        table_name=LLAMAINDEX_PGVECTOR_COLLECTION,
        embed_dim=EMBED_DIM, # nomic-embed-text is 768? depends on model; must match Settings.embed_model dim
        # optionally schema_name="public"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=pgvs
    )

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        embed_model=embed_model
        )


if __name__ == "__main__":
    ingest()

    ### Find out the number of dimensions
    # embed_model = OllamaEmbedding(
    #     model_name=OLLAMA_EMBED_MODEL
    # )

    # v = embed_model.get_text_embedding("dimension check")
    # embed_dim = len(v)
    # print("embed_dim:", embed_dim)
