"""
Stage 2: Storage Layer
Manages relational storage (SQLite via SQLAlchemy) and columnar storage (Parquet / DuckDB)
for high-performance rolling aggregation queries.
"""

import sqlite3
import pandas as pd
import duckdb
from pathlib import Path
from typing import Optional
from mlb_engine.config import DB_PATH, PARQUET_DIR

class MLBDataStorage:
    """Storage manager supporting relational SQLite queries and fast columnar Parquet/DuckDB operations."""

    def __init__(self, db_path: Path = DB_PATH, parquet_dir: Path = PARQUET_DIR):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self._init_sqlite_db()

    def _init_sqlite_db(self):
        """Initializes SQLite tables for game logs, pitch data, and lineup metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_starter_id INT,
                away_starter_id INT,
                home_score INT,
                away_score INT,
                total_runs INT,
                venue TEXT,
                temperature REAL,
                wind_speed REAL,
                wind_direction TEXT,
                umpire_home TEXT
            )
        """)

        # Statcast pitch-level summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pitch_events (
                pitch_id TEXT PRIMARY KEY,
                game_id TEXT,
                game_date TEXT,
                pitcher_id INT,
                batter_id INT,
                pitch_type TEXT,
                release_speed REAL,
                events TEXT,
                description TEXT,
                estimated_woba_using_speedangle REAL,
                launch_speed REAL,
                launch_angle REAL,
                is_strikeout INT,
                is_walk INT,
                is_swinging_strike INT
            )
        """)

        # Player rolling daily metrics cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_daily_stats (
                player_id INT,
                date TEXT,
                role TEXT,
                k_pct REAL,
                bb_pct REAL,
                k_bb_diff REAL,
                siera REAL,
                fip REAL,
                xfip REAL,
                whiff_pct REAL,
                hard_hit_pct REAL,
                woba REAL,
                wrc_plus REAL,
                PRIMARY KEY (player_id, date, role)
            )
        """)

        conn.commit()
        conn.close()

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """Saves a pandas DataFrame into SQLite relational database."""
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()

    def save_parquet(self, df: pd.DataFrame, file_name: str):
        """Saves DataFrame as Parquet columnar format for fast memory-mapped analytical reads."""
        path = self.parquet_dir / f"{file_name}.parquet"
        df.to_parquet(path, index=False)

    def read_query_sql(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Executes a SQL query against SQLite."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def query_parquet_duckdb(self, query: str) -> pd.DataFrame:
        """Executes high-speed DuckDB SQL analytical queries directly on Parquet files."""
        con = duckdb.connect()
        # Allows using 'parquet_dir/*.parquet' inside queries
        parquet_path = str(self.parquet_dir / "*.parquet")
        query_resolved = query.replace("PARQUET_FILES", f"'{parquet_path}'")
        df = con.execute(query_resolved).df()
        con.close()
        return df
