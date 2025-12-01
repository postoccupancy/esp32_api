from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
from supabase import create_client, Client  # pip install supabase
import uvicorn
import time

# 1) Pydantic model (FastAPI docs: "Tutorial » Request Body")
# Define the expected structure of incoming JSON data
class Reading(BaseModel):
    device_id: str = Field(..., example="esp32-s3-devkit-001")
    timestamp: float = Field(default_factory=lambda: time.time())
    temperature_c: float | None = None
    humidity: float | None = None
    rms: float | None = None

# 2) Supabase client (FastAPI docs: "Settings and Environment Variables")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # server-side only!
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="ESP Ingest")

@app.get("/")
def root():
    return {"ok": True, "service": "ingest"}

# 3) Ingest endpoint (FastAPI docs: "Tutorial » Request Body")
@app.post("/ingest")
async def ingest(r: Reading):
    try:
        data = r.dict()
        resp = sb.table("sensor_events").insert(data).execute()
        return {"ok": True, "inserted": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # FastAPI docs: "Deployment" (dev server via uvicorn)
    uvicorn.run(app, host="0.0.0.0", port=8000)
