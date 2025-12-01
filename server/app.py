# app.py
import os, sqlite3, time
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()  # take environment variables from .env file

from supabase import create_client, Client  # pip install supabase
import uvicorn

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Accept either service role or anon; prefer service role on the server.
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "readings")
SHARED_TOKEN = os.getenv("INGEST_TOKEN", "change-me")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[supabase] client initialized")
    except Exception as e:
        print("[supabase] init failed:", repr(e))
        supabase = None
else:
    print("[supabase] missing SUPABASE_URL or SUPABASE_*KEY; skipping client")

app = FastAPI()

def insert_supabase(row: dict):
    if supabase is None:
        return None
    try:
        # send as a list for maximum compatibility
        res = supabase.table(SUPABASE_TABLE).insert([row]).execute()
        # Supabase-py v2 returns a Postgres response with .data
        print("[supabase] insert data:", getattr(res, "data", None))
        return res
    except Exception as e:
        print("[supabase] insert error:", repr(e))
        return None

latest_reading = None

@app.get("/ping")
def ping():
    return {"pong": True, "sqlite": True, "supabase": bool(supabase)}

@app.get("/latest")
def get_latest():
    return latest_reading or {}

@app.post("/ingest")
async def ingest(request: Request, x_token: str = Header(None)):
    global latest_reading
    if x_token != SHARED_TOKEN:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    data = await request.json()
    data["ts"] = datetime.now(timezone.utc).isoformat()
    
    # Basic shape guard (keep it loose for now)
    required = {"device_id", "ts"}
    if not required.issubset(set(data.keys())):
        return JSONResponse({"ok": False, "error": "missing device_id or ts"}, status_code=400)

    # Log to console
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), data)

    # Supabase write (if configured)
    sb_status = None
    if supabase:
        sb_res = insert_supabase(data)
        sb_status = "ok" if sb_res and getattr(sb_res, "data", None) else "error"

    latest_reading = data

    return {"ok": True, "supabase": sb_status}

if __name__ == "__main__":
    import uvicorn
    # LAN-exposed so ESP32 can reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)
