import os
from datetime import datetime, timezone, timedelta
import pytz
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "readings")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4", "")

# supabase = None
# if SUPABASE_URL and SUPABASE_KEY:
#     try:
#         from supabase import create_client
#         supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
#         print("[supabase] client initialized")
#     except Exception as e:
#         print("[supabase] init failed:", repr(e))
#         supabase = None
# else:
#     print("[supabase] missing SUPABASE_URL or SUPABASE_*KEY; skipping client")


# query the Supabase DB
import psycopg2
from datetime import datetime, timezone

DEVICE_ID="esp32-s3-devkit-001"

def get_readings_since(device_id, hours=24):
    connection = psycopg2.connect(SUPABASE_DB_URL)
    cursor = connection.cursor()

    query = f"""
        SELECT
            ts::timestamptz,
            temp_c,
            temp_f,
            rh
        FROM {SUPABASE_TABLE}
        WHERE device_id = %s
        AND ts::timestamptz >= now() - INTERVAL '%s hours'
        ORDER BY ts::timestamptz ASC;
    """
    cursor.execute(query, (
        device_id, 
        hours
        ))
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return rows

def build_snapshot(device_id: str, rows):
    if not rows: 
        return None
    
    tempsc = [r[1] for r in rows if r[1] is not None]
    tempsf = [r[2] for r in rows if r[2] is not None]
    humdts = [r[3] for r in rows if r[3] is not None]

    times = [r[0] for r in rows if r[0] is not None]
    window_start = times[0]
    window_end = times[-1]
    interval = window_end - window_start
    # now = datetime.now(timezone.utc)
    # since = now - window_end
    number = len(times)

    snapshot_text = f"""
Sensor: {device_id}
Window: {(interval.total_seconds() / 3600):.2f} hours
Start: {window_start.isoformat()}
End: {window_end.isoformat()}
Data coverage: {(number * 100) / (interval.total_seconds() / 2):.0f}%

Temperature: 
- avg: {sum(tempsc)/len(tempsc):.2f}°C / {sum(tempsf)/len(tempsf):.2f}°F
- min: {min(tempsc):.2f}°C / {min(tempsf):.2f}°F 
- max: {max(tempsc):.2f}°C / {max(tempsf):.2f}°F

Humidity:
- avg: {sum(humdts)/len(humdts):.2f}%
- min: {min(humdts):.2f}%
- max: {max(humdts):.2f}%
"""
    print(snapshot_text)
    return snapshot_text, window_end, window_start

rows = get_readings_since(DEVICE_ID, hours=6)
build_snapshot(DEVICE_ID, rows)

if __name__ == "_main__":
    build_snapshot(DEVICE_ID, rows)