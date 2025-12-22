from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Dict, Any

import psycopg  # psycopg v3, not psycopg2

from langchain_core.documents import Document

import os
DEVICE_ID = os.getenv("DEVICE_ID")
RAW_DATA_TABLE = os.getenv("RAW_DATA_TABLE")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL_IPV4")


@dataclass(frozen=True)
class SnapshotWindow:
    device_id: str
    window_start: datetime
    window_end: datetime

@dataclass(frozen=True)
class HourlySnapshot:
    device_id: str
    window_start: datetime
    window_end: datetime
    n: int
    t_min_c: float
    t_max_c: float
    t_avg_c: float
    t_min_f: float
    t_max_f: float
    t_avg_f: float
    h_min: float
    h_max: float
    h_avg: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _get_earliest_timestamp(
    connection: psycopg.Connection,
    table_name: str = RAW_DATA_TABLE,
    timestamp_column_name: str = "ts",
    device_id: Optional[str] = None,        
) -> Optional[datetime]:
    if device_id is None:
        sql = f"""
            select min({timestamp_column_name}) 
            from {table_name};
        """
        params = None
    else:
        sql = f"""
            select min({timestamp_column_name})
            from {table_name}
            where device_id = %(device_ide)s;
        """
        params = {"device_id": device_id}

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        earliest = cursor.fetchone()[0]

    if earliest is None:
        return None
    
    # Normalize to UTC timezone-aware datetime
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    else:
        earliest = earliest.astimezone(timezone.utc)

    return earliest        



def _define_hour_windows_from_start(
        device_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
) -> List[SnapshotWindow]:

    window_start: Optional[datetime] = None
    if start_time:
        window_start = _floor_to_hour(start_time)
    else:
        connection = psycopg.connect(SUPABASE_DB_URL)
        earliest = _get_earliest_timestamp(
            connection=connection,
            timestamp_column_name="created_at"
        )
        window_start = _floor_to_hour(earliest)

    window_end = _floor_to_hour(end_time or _utc_now())

    print(f"Defining hour windows from {window_start} to {window_end}")

    windows: List[SnapshotWindow] = []
    current_dt = window_start
    while current_dt < window_end:
        snapshot = SnapshotWindow(
                device_id=device_id,
                window_start=current_dt,
                window_end=current_dt + timedelta(hours=1),
            )
        windows.append(snapshot)
        print(f"{snapshot.device_id}: {snapshot.window_start.isoformat()} to {snapshot.window_end.isoformat()}")
        current_dt += timedelta(hours=1)

    return windows


def _define_hour_windows_from_end(
    device_id: str,
    lookback_hours: int,
    end_time: Optional[datetime] = None,
) -> List[SnapshotWindow]:
    window_end = _floor_to_hour(end_time or _utc_now())
    window_start = window_end - timedelta(hours=lookback_hours)

    print(f"Defining hour windows looking back {lookback_hours} hours from {window_end}...")

    windows: List[SnapshotWindow] = []
    current_dt = window_start
    while current_dt < window_end:
        snapshot = SnapshotWindow(
            device_id=device_id,
            window_start=current_dt,
            window_end=current_dt + timedelta(hours=1),
            )
        windows.append(snapshot)
        print(f"{snapshot.device_id}: {snapshot.window_start.isoformat()} to {snapshot.window_end.isoformat()}")
        current_dt += timedelta(hours=1)
    """
    Returns a list of windows:
      [{"device_id": ..., "window_start": ..., "window_end": ...}, ...]
    """
    return windows




def _fetch_hour_stats(
    connection: psycopg.Connection,
    window: SnapshotWindow,
    timestamp_column_name: str = "ts",
    temperature_c_column_name: str = "temp_c",
    temperature_f_column_name: str = "temp_f",
    humidity_column_name: str = "rh",
) -> Optional[HourlySnapshot]:
    """
    Returns None if no data in that hour window.
    """
    sql = f"""
        select
          count(*) as n,
          min({temperature_c_column_name}) as t_min_c,
          max({temperature_c_column_name}) as t_max_c,
          avg({temperature_c_column_name}) as t_avg_c,
          min({temperature_f_column_name}) as t_min_f,
          max({temperature_f_column_name}) as t_max_f,
          avg({temperature_f_column_name}) as t_avg_f,
          min({humidity_column_name}) as h_min,
          max({humidity_column_name}) as h_max,
          avg({humidity_column_name}) as h_avg
        from {RAW_DATA_TABLE}
        where device_id = %(device_id)s
          and {timestamp_column_name} >= %(start)s
          and {timestamp_column_name} <  %(end)s;
    """

    # for psycopg v3, the %s placeholders won't work if you pass in the SnapshotWindow dataclass
    # so we need to unpack it into a dict
    params = {
        "device_id": window.device_id,
        "start": window.window_start,
        "end": window.window_end,
    }

    cursor = connection.cursor()
    cursor.execute(sql, params)
    row = cursor.fetchone()
    cursor.close()

    count = int(row[0])

    if not row or count == 0:
        return None

    return HourlySnapshot(
        device_id=window.device_id,
        window_start=window.window_start,
        window_end=window.window_end,
        n=int(count),
        t_min_c=float(row[1]),
        t_max_c=float(row[2]),
        t_avg_c=float(row[3]),
        t_min_f=float(row[4]),
        t_max_f=float(row[5]),
        t_avg_f=float(row[6]),
        h_min=float(row[7]),
        h_max=float(row[8]),
        h_avg=float(row[9]),
    )


def _format_snapshot_text(
    window: SnapshotWindow,
    stats: HourlySnapshot,
) -> str:
    return (
        f"Hourly snapshot for device '{window.device_id}'\n"
        f"Window (UTC): {window.window_start.isoformat()} → {window.window_end.isoformat()}\n"
        f"Samples: {stats.n}\n"
        f"Data coverage: {((stats.n * 100) / 1800):.2f }%\n"
        f"Temperature °C: avg={stats.t_avg_c:.2f}, "
        f"min={stats.t_min_c:.2f}, max={stats.t_max_c:.2f}\n"
        f"Humidity %RH: avg={stats.h_avg:.2f}, "
        f"min={stats.h_min:.2f}, max={stats.h_max:.2f}\n"
    )


def build_langchain_documents(
    lookback_hours: Optional[int] = None,
    end_time: Optional[datetime] = None,
    start_time: Optional[datetime] = None,        
) -> List[Document]:
    """
    Connects to Supabase Postgres, queries raw readings, and returns hourly Documents using all functions above.
    This will be called in rag_index.py to index the hourly snapshots.
    """

    connection = psycopg.connect(SUPABASE_DB_URL)

    documents: List[Document] = []

    hours: List[SnapshotWindow] = []

    if lookback_hours:
        hours = _define_hour_windows_from_end(
            DEVICE_ID, 
            lookback_hours=lookback_hours, 
            end_time=end_time
        )
    else:    
        hours = _define_hour_windows_from_start(
            DEVICE_ID,
            start_time=start_time,
            end_time=end_time
        )


    try:
        for hour in hours:
            window_start: datetime = hour.window_start
            window_end: datetime = hour.window_end

            stats = _fetch_hour_stats(connection, hour)
            if stats is None:
                continue

            snapshot_text = _format_snapshot_text(hour, stats)

            documents.append(
                Document(
                    page_content=snapshot_text, 
                    metadata={
                        "window_start": window_start.isoformat(), 
                        "window_end": window_end.isoformat(),
                        "kind": "hourly_snapshot",
                        "device_id": DEVICE_ID,
                        "source": RAW_DATA_TABLE,
                        "data_coverage": stats.n / 1800
                       }
                    ))
    finally:
        connection.close()

    return documents



if __name__ == "__main__":
    documents = build_langchain_documents(lookback_hours=2)
    for doc in documents:
        print(doc.page_content)