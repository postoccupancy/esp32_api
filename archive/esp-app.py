from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import uvicorn, time

app = FastAPI()

SHARED_TOKEN = "change-me"  # simple shared secret

@app.post("/ingest")
async def ingest(request: Request, x_token: str = Header(None)):
    if x_token != SHARED_TOKEN:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    data = await request.json()
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), data)
    return {"ok": True}

if __name__ == "__main__":
    # 0.0.0.0 exposes on LAN so ESP32 can reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)
