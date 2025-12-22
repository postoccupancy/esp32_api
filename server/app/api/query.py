from app.api.rag_router import rag_query

def query(query: str):
    json = {
        "question": query
    }
    response = rag_query(query)
    return response

if __name__ == "__main__":
    query("what can you do?")