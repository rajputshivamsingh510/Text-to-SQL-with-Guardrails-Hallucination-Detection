from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.db import make_engine
from app.demo_data import seed
from app.pipeline import Pipeline
from app.prompts import JUDGE_SYSTEM, SUMMARY_SYSTEM
from app.retriever import SchemaRetriever, load_examples
from app.schema import load_schema


class FakeLLM:
    """Scripted LLM. SQL calls consume `sql` in order (the last item repeats); judge/summary are fixed."""

    def __init__(self, sql=(), judge=None, summary="There are results.", fail=False):
        self.sql = list(sql)
        self.judge = judge if judge is not None else {"answers_question": True, "score": 0.95, "issues": []}
        self.summary = summary
        self.fail = fail
        self.calls = {"sql": 0, "judge": 0, "summary": 0}

    def complete(self, system, user, *, temperature=0.0, max_tokens=1024):
        if self.fail:
            raise RuntimeError("boom")
        if system == JUDGE_SYSTEM:
            self.calls["judge"] += 1
            return json.dumps(self.judge)
        if system == SUMMARY_SYSTEM:
            self.calls["summary"] += 1
            return self.summary
        i = self.calls["sql"]
        self.calls["sql"] += 1
        item = self.sql[min(i, len(self.sql) - 1)]
        return item if item.startswith("<") else f"<sql>{item}</sql>"

    @property
    def total(self):
        return sum(self.calls.values())


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "shop.db"
    seed(make_engine(f"sqlite:///{path}", read_only=False))
    return path


@pytest.fixture()
def settings(db_path):
    return Settings(database_url=f"sqlite:///{db_path}", blocked_columns=("customers.email", "customers.phone"))


@pytest.fixture()
def engine(settings):
    return make_engine(settings.database_url, read_only=True)


@pytest.fixture()
def schema(engine, settings):
    return load_schema(engine, settings)


@pytest.fixture()
def make_pipeline(settings, engine, schema):
    def _make(llm, **overrides):
        s = Settings(**{**settings.__dict__, **overrides})
        retriever = SchemaRetriever(schema, load_examples(s.data_dir / "examples.json"))
        return Pipeline(s, engine, schema, retriever, llm)

    return _make
