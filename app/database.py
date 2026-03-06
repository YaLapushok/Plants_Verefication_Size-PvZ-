"""
database.py – SQLite storage for plant analysis results.
Tables:
  analyses(id, timestamp, filename, plant_class, seg_model,
           total_length, total_area, class_metrics_json,
           original_bytes, segmented_bytes)
"""

import sqlite3
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH  = os.path.join(DATA_DIR, 'analyses.db')


def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                plant_class     TEXT    NOT NULL,
                seg_model       TEXT    NOT NULL DEFAULT 'yolo',
                total_length    REAL    NOT NULL DEFAULT 0,
                total_area      REAL    NOT NULL DEFAULT 0,
                class_metrics_json TEXT NOT NULL DEFAULT '{}',
                original_bytes  BLOB,
                segmented_bytes BLOB
            )
        """)
        con.commit()


def save_analysis(
    filename: str,
    plant_class: str,
    seg_model: str,
    total_length: float,
    total_area: float,
    class_metrics: dict,
    original_bytes: bytes,
    segmented_bytes: bytes,
) -> int:
    init_db()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO analyses
               (timestamp, filename, plant_class, seg_model,
                total_length, total_area, class_metrics_json,
                original_bytes, segmented_bytes)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts, filename, plant_class, seg_model,
             total_length, total_area,
             json.dumps(class_metrics, ensure_ascii=False),
             original_bytes, segmented_bytes),
        )
        con.commit()
        return cur.lastrowid


def list_analyses() -> list:
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT id, timestamp, filename, plant_class, seg_model,
                      total_length, total_area, class_metrics_json
               FROM analyses ORDER BY id DESC"""
        ).fetchall()
    result = []
    for r in rows:
        item = dict(r)
        item['class_metrics'] = json.loads(item.pop('class_metrics_json', '{}'))
        result.append(item)
    return result


def get_analysis(analysis_id: int) -> dict | None:
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM analyses WHERE id=?", (analysis_id,)
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item['class_metrics'] = json.loads(item.pop('class_metrics_json', '{}'))
    return item


def delete_analysis(analysis_id: int) -> bool:
    init_db()
    with _conn() as con:
        cur = con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
        con.commit()
        return cur.rowcount > 0

def delete_all_analyses() -> bool:
    init_db()
    with _conn() as con:
        cur = con.execute("DELETE FROM analyses")
        con.commit()
        return cur.rowcount > 0
