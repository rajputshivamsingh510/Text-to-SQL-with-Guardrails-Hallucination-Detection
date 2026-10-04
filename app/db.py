"""Database access. Everything the pipeline runs goes through run_select() on a read-only engine."""
from __future__ import annotations

import datetime as dt
import decimal
import json
from dataclasses import dataclass

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from app.config import normalize_db_url


def make_engine(url: str, *, read_only: bool, statement_timeout_ms: int = 5000) -> Engine:
    url = normalize_db_url(url)
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})
        if read_only:

            @event.listens_for(engine, "connect")
            def _sqlite_read_only(dbapi_conn, _record):
                dbapi_conn.execute("PRAGMA query_only = ON")

        return engine

    connect_args: dict = {}
    if read_only:
        # Session-level safety net, on top of the read-only DB role you create for production.
        connect_args["options"] = (
            f"-c statement_timeout={int(statement_timeout_ms)} -c default_transaction_read_only=on"
        )
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args=connect_args)


def sqlglot_dialect(engine: Engine) -> str:
    return {"postgresql": "postgres", "sqlite": "sqlite", "mysql": "mysql"}.get(engine.dialect.name, "postgres")


def _escape(sql: str) -> str:
    # text() treats ":name" as a bind parameter; escape colons so string literals survive.
    return sql.replace(":", "\\:")


def _json_safe(value):
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary>"
    return value


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool

    @property
    def row_count(self) -> int:
        return len(self.rows)


def run_select(engine: Engine, sql: str, max_rows: int) -> QueryResult:
    with engine.connect() as conn:
        result = conn.execute(text(_escape(sql)))
        columns = list(result.keys())
        fetched = result.fetchmany(max_rows + 1)
    truncated = len(fetched) > max_rows
    rows = [[_json_safe(v) for v in row] for row in fetched[:max_rows]]
    return QueryResult(columns=columns, rows=rows, truncated=truncated)


def estimate_cost(engine: Engine, sql: str) -> float | None:
    """Postgres only: planner cost estimate. Also surfaces type/semantic errors without executing."""
    if engine.dialect.name != "postgresql":
        return None
    with engine.connect() as conn:
        row = conn.execute(text("EXPLAIN (FORMAT JSON) " + _escape(sql))).fetchone()
    plan = row[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    return float(plan[0]["Plan"]["Total Cost"])


def value_exists(engine: Engine, table: str, column: str, value, *, case_insensitive: bool = False) -> bool:
    """Does `value` occur in table.column? Identifiers must already be validated against the schema."""
    q = engine.dialect.identifier_preparer.quote
    if case_insensitive:
        stmt = text(f"SELECT 1 FROM {q(table)} WHERE LOWER(CAST({q(column)} AS TEXT)) = LOWER(:v) LIMIT 1")
    else:
        stmt = text(f"SELECT 1 FROM {q(table)} WHERE {q(column)} = :v LIMIT 1")
    with engine.connect() as conn:
        return conn.execute(stmt, {"v": value}).first() is not None
