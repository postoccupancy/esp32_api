# **ESP32 Sensor API & Chat Agent**

A data ingestion, retrieval, and analysis backend for microcontroller sensor streams (ESP32) with initial support for RAG-style natural language querying.  
 This repository provides:

* a FastAPI server for ingesting and querying sensor data

* tools for semantic indexing and retrieval via LLM agent

* a planned UI component (Next.js frontend integration) for data visualization and interaction

This project is evolving toward a production-ready sensor data platform with LLM chat interaction and analytics.

---

## **Overview**

The `esp32_api` project is designed to collect, store, and query time-series sensor data from ESP32 microcontrollers. It also integrates an agentic AI framework that allows users to ask open-ended questions about trends and anomalies in the data, experimenting with both Retrieval Augmented Generation (RAG) and natural language-to-SQL (NLSQL) strategies.

Key aspects of the stack include:

* FastAPI backend with REST endpoints (`/ingest`, `/timeseries`)

* PostgreSQL database for sensor storage and snapshots

* Vector embedding and semantic retrieval endpoints (`/rag/query`, `/rag/index`, `/rag/ingest_docs`)

* Planned integration with a React/Next.js frontend for visualization and agent interaction

---

## **Project Structure**

`esp32_api/`  
`├── server/                  # Backend service (FastAPI)`  
`│   ├── app/                # Core application modules`  
`│   ├── rag/                # RAG & LLM logic`  
`│   └── main.py             # Data ingestion & status`  
`├── device/                  # MicroPython scripts for ESP32`  
`├── ui/                      # Placeholder for future frontend`  
`├── docs/                    # Project documentation`  
`├── .env.example             # Example environment vars`  
`└── README.md                # This file`


---

## **API Endpoints**

### **Ingestion**

`POST /ingest`  
 Ingests sensor payloads from ESP32 devices. Payload structure and parameters TBD.

### **Time-Series Queries**

`GET /timeseries`  
Returns filtered time-series data based on query parameters (`sensor`, `from`, `to`, `avg`, `min`, `max`, etc.).

### **RAG & Semantic Endpoints**

* `POST /rag/query` — Chatbot that answers questions by generating SQL queries for the data and searching contextual documents.

* `POST /rag/index` — Batch embedding of time-series snapshots into vector store so LLM can answer data questions without SQL.

* `POST /rag/ingest_docs` — Splits up PDFs and web pages into small, overlapping text chunks and embeds them in a vector DB.

*(Detailed request/response schemas to be documented in `/docs/endpoints/*.md`.)*

---

## **Architecture**

The system is composed of the following high-level components:

`ESP32 Microcontroller`  
      `↓ (posts data live via HTTP)`  
`FastAPI Backend ── PostgreSQL ── Snapshots & Raw Data`  
      `├─ /ingest, /timeseries`  
      `└─ /rag/index, /rag/query, /rag/ingest_docs`  
           `├─ SQL + Time-Series Logic`  
           `└─ RAG / Embeddings`  
`Front-end (Next.js / React / TypeScript) — visualization & agent interaction`

The planned frontend may integrate the Vercel/Next AI SDK, which would move the /rag endpoints to Front-end.

---

## **Installation**

### **Prerequisites**

* **Python 3.11+**

* **PostgreSQL** instance with credentials available (e.g. Supabase)

* `pgvector` extension installed in PostgreSQL for vector embeddings (e.g. in Supabase, install `vector` under Database > Extensions)

### **Local Setup**

Clone the repository and install dependencies:

`git clone https://github.com/postoccupancy/esp32_api.git`  
`cd esp32_api`  
`pip install -r server/requirements.txt`

Set up environment variables (see **Configuration** below), then start the API:

`uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload`

---


## **Configuration**

Environment variables are used to control database connections. Copy `.env.example` to `.env` and fill in the required values.


---


## **Documentation Structure**

This repository follows a structured documentation layout inspired by best practices. The primary documentation is housed under the `docs/` folder. Key sections include:

**Notes & Tutorials**

* [`docs/notes/2025-12-17-open-source-agent-stack.md`](/blob/main/docs/2025-12-17-open-source-agent-stack.md) — Mapping out a free / open source agent development stack. 

* [`2026-01-27-rag-setup-and-next-steps.md`](/blob/main/docs/2026-01-27-rag-setup-and-next-steps) — Discussion of how the RAG endpoints work, and how I plan to use them in an AI-enhanced data visualization interface. 

* *(Future)* additional notes covering architectural decisions and integrations.


---

## **Contributing**

Contributions are welcome\! For structured guidelines, see the `CONTRIBUTING.md` once created. For now:

1. Fork the repo

2. Create a descriptive branch

3. Open a pull request with context and tests (when available)

---

## **License**

This project is open source and released under the BSD-3 Clause License.

---

## **What’s Next / Roadmap**

More documentation planned...

