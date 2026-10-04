"""FastAPI service. Start with:  uvicorn app.main:app --host 0.0.0.0 --port $PORT"""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from sqlalchemy import text

from app.config import Settings
from app.db import make_engine
from app.demo_data import has_data, seed
from app.llm import LLM, AnthropicLLM
from app.models import QueryRequest, QueryResponse
from app.pipeline import Pipeline
from app.retriever import SchemaRetriever, load_examples
from app.schema import load_schema
from app.llm import LLM, build_llm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("text2sql")


def build_pipeline(settings: Settings, llm: LLM | None) -> Pipeline:
    if settings.auto_seed_demo:
        admin = make_engine(settings.effective_admin_url, read_only=False)
        if not has_data(admin):
            log.info("AUTO_SEED_DEMO: seeding demo data")
            seed(admin)
        admin.dispose()
    engine = make_engine(settings.database_url, read_only=True, statement_timeout_ms=settings.statement_timeout_ms)
    schema = load_schema(engine, settings)
    retriever = SchemaRetriever(schema, load_examples(settings.data_dir / "examples.json"))
    return Pipeline(settings, engine, schema, retriever, llm)  # type: ignore[arg-type]


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        the_llm = llm if llm is not None else build_llm(settings)
        if the_llm is None:
            log.warning("No API key set for LLM_PROVIDER=%s (GROQ_API_KEY): /query will return 503 until one is configured.",
                        settings.llm_provider)
        app.state.pipeline = build_pipeline(settings, the_llm)
        app.state.llm_ready = the_llm is not None
        yield
        app.state.pipeline.engine.dispose()

    app = FastAPI(title="Text-to-SQL with Guardrails & Hallucination Detection", version="1.0.0", lifespan=lifespan)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")

    @app.get("/health")
    def health():
        try:
            with app.state.pipeline.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail="database unavailable")
        return {"status": "ok", "llm_configured": app.state.llm_ready}

    @app.get("/schema", dependencies=[Depends(require_key)])
    def get_schema():
        """The tables and columns the assistant is allowed to see (hidden columns are not listed)."""
        return {name: [c.name for c in t.columns] for name, t in app.state.pipeline.schema.tables.items()}

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_key)])
    def query(req: QueryRequest):
        if not app.state.llm_ready:
            raise HTTPException(status_code=503, detail="LLM is not configured (set GROQ_API_KEY).")
        return app.state.pipeline.run(req.question)

    return app


app = create_app()
