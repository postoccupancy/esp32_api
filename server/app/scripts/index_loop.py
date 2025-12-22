import time
import requests

from app.api.rag_router import rag_index

TOKEN = "a-long-secret-key"

# index every hour while app is running
# index endpoint flattens to the top of the current hour and takes the last hour
# so at 4:15 it will index from 3:00 to 4:00

print("Starting indexing loop...")
def index_loop():
    while True:
        result = rag_index()
        print(result)
        time.sleep(3600)  # one hour


if __name__ == "__main__":
    index_loop()