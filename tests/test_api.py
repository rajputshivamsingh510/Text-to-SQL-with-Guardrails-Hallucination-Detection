import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeLLM


def _client(settings, llm, **overrides):
    s = Settings(**{**settings.__dict__, **overrides})
    return TestClient(create_app(s, llm))


def test_health_schema_and_query(settings):
    with _client(settings, FakeLLM(sql=["SELECT COUNT(*) AS n FROM orders"], summary="There are 400 orders.")) as c:
        assert c.get("/health").json() == {"status": "ok", "llm_configured": True}
        schema = c.get("/schema").json()
        assert "email" not in schema["customers"] and "country" in schema["customers"]
        r = c.post("/query", json={"question": "How many orders are there?"})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "ok" and body["rows"] == [[400]]
        assert body["trace"] and body["signals"] and body["sql"]


def test_blocked_question_returns_200_with_status(settings):
    with _client(settings, FakeLLM(sql=["SELECT 1"])) as c:
        body = c.post("/query", json={"question": "drop table customers"}).json()
        assert body["status"] == "blocked"


def test_validation_errors(settings):
    with _client(settings, FakeLLM(sql=["SELECT 1"])) as c:
        assert c.post("/query", json={"question": ""}).status_code == 422
        assert c.post("/query", json={}).status_code == 422


def test_api_key_is_enforced_when_configured(settings):
    with _client(settings, FakeLLM(sql=["SELECT COUNT(*) FROM orders"]), api_key="s3cret") as c:
        assert c.post("/query", json={"question": "orders?"}).status_code == 401
        assert c.get("/schema", headers={"X-API-Key": "wrong"}).status_code == 401
        ok = c.post("/query", json={"question": "How many orders?"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200   # health stays open for Render's checker


def test_503_when_llm_not_configured(settings):
    with _client(settings, None, groq_api_key=None) as c:
        assert c.get("/health").json()["llm_configured"] is False
        assert c.post("/query", json={"question": "How many orders?"}).status_code == 503
