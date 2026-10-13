"""
db/warehouse.py
---------------
ALL database code lives in this one module so that switching from SQLite
to MySQL later only means changing get_connection() (and a few SQL dialect
details such as AUTOINCREMENT -> AUTO_INCREMENT).

This step: connection + saved assessments.
Later step: star schema, ETL and OLAP queries are added here.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agrimine.db"


def get_connection() -> sqlite3.Connection:
    """Single place that opens the database. Swap this for MySQL later."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create tables if they do not exist yet (safe to call on every run)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT NOT NULL,
                N              REAL, P REAL, K REAL,
                ph             REAL, temperature REAL,
                humidity       REAL, rainfall REAL,
                predicted_crop TEXT,   -- filled in by the Crop Advisor page
                cluster_id     INTEGER -- filled in by the Cluster Explorer page
            )
            """
        )


def save_assessment(values: dict, predicted_crop: str | None = None,
                    cluster_id: int | None = None) -> int:
    """Insert one soil assessment and return its id."""
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO assessments
                (created_at, N, P, K, ph, temperature, humidity, rainfall,
                 predicted_crop, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().strftime("%Y-%m-%d %H:%M"),
             values["N"], values["P"], values["K"], values["ph"],
             values["temperature"], values["humidity"], values["rainfall"],
             predicted_crop, cluster_id),
        )
        return cur.lastrowid


def get_recent_assessments(limit: int = 10) -> pd.DataFrame:
    """Return the most recent saved assessments (newest first)."""
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
