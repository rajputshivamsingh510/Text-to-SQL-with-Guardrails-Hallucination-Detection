"""Runs the real pipeline against Postgres using the least-privilege role. Skipped unless configured:

    TEST_PG_RO_URL=postgresql://t2sql_readonly:...@localhost/shop pytest tests/test_postgres_integration.py
"""
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings
from app.db import make_engine, run_select
from app.pipeline import Pipeline
from app.retriever import SchemaRetriever, load_examples
from app.schema import load_schema
from tests.conftest import FakeLLM

RO_URL = os.environ.get("TEST_PG_RO_URL")
pytestmark = pytest.mark.skipif(not RO_URL, reason="TEST_PG_RO_URL not set")

BLOCKED = ("customers.email", "customers.phone")


@pytest.fixture(scope="module")
def pg():
    settings = Settings(database_url=RO_URL, blocked_columns=BLOCKED, statement_timeout_ms=2000)
    engine = make_engine(RO_URL, read_only=True, statement_timeout_ms=2000)
    schema = load_schema(engine, settings)
    retriever = SchemaRetriever(schema, load_examples(settings.data_dir / "examples.json"))
    return settings, engine, schema, retriever


def _run(pg, llm, question):
    settings, engine, schema, retriever = pg
    return Pipeline(settings, engine, schema, retriever, llm).run(question)


def test_database_itself_refuses_writes_pii_and_slow_queries(pg):
    _, engine, _, _ = pg
    for sql, msg in [("DELETE FROM orders", "read-only"), ("DROP TABLE orders", "read-only"),
                     ("SELECT email FROM customers", "permission denied"),
                     ("SELECT pg_sleep(10)", "statement timeout")]:
        with pytest.raises(SQLAlchemyError, match=msg):
            run_select(engine, sql, 5)


def test_pipeline_end_to_end_on_postgres(pg):
    llm = FakeLLM(sql=["SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC"],
                  summary="Done.")
    resp = _run(pg, llm, "How many customers per country?")
    assert resp.status == "ok" and resp.row_count == 6


def test_select_star_works_because_guard_expands_it(pg):
    resp = _run(pg, FakeLLM(sql=["SELECT * FROM customers"], summary="Done."), "Show all customers")
    assert resp.status == "ok" and "email" not in resp.columns


def test_value_grounding_on_postgres(pg):
    llm = FakeLLM(sql=["SELECT COUNT(*) FROM customers WHERE country = 'USA'",
                       "SELECT COUNT(*) FROM customers WHERE country = 'United States'"])
    resp = _run(pg, llm, "How many customers are in the USA?")
    assert resp.status == "ok" and resp.attempts == 2 and "United States" in resp.sql


def test_guard_rejects_postgres_specific_attacks(pg):
    llm = FakeLLM(sql=["SELECT pg_sleep(10)", "SELECT * FROM pg_catalog.pg_shadow", "SELECT * FROM customers FOR UPDATE"])
    resp = _run(pg, llm, "How many customers do we have?")
    assert resp.status == "rejected" and resp.rows == []
