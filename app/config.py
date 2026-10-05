"""Central configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent


def normalize_db_url(url: str) -> str:
    """Render gives postgres:// or postgresql:// URLs; SQLAlchemy needs a driver suffix."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(value: str | None) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in (value or "").split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = "sqlite:///./demo.db"
    admin_database_url: str = ""  # used only for seeding; falls back to database_url
    allowed_tables: tuple[str, ...] = ()  # empty = every table
    blocked_columns: tuple[str, ...] = ()  # "table.column": hidden from the LLM and from queries
    max_rows: int = 200
    statement_timeout_ms: int = 5000
    max_query_cost: float = 1_000_000.0  # Postgres EXPLAIN cost ceiling
    auto_seed_demo: bool = False
    sample_values: bool = True  # show the LLM the distinct values of low-cardinality text columns

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = None

    # Guardrails
    max_question_chars: int = 500
    pii_mode: str = "mask"  # mask | block | off
    max_retries: int = 2
    confidence_threshold: float = 0.6
    self_consistency_samples: int = 0  # extra SQL samples used to cross-check; costs extra LLM calls
    generate_summary: bool = True

    # API
    api_key: str | None = None  # if set, clients must send X-API-Key

    data_dir: Path = ROOT / "data"

    @property
    def effective_admin_url(self) -> str:
        return self.admin_database_url or self.database_url

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")  # real environment variables win over .env (override=False)
        e = os.environ.get
        return cls(
            database_url=e("DATABASE_URL", cls.database_url),
            admin_database_url=e("ADMIN_DATABASE_URL", ""),
            allowed_tables=_list(e("ALLOWED_TABLES")),
            blocked_columns=_list(e("BLOCKED_COLUMNS")),
            max_rows=int(e("MAX_ROWS", "200")),
            statement_timeout_ms=int(e("STATEMENT_TIMEOUT_MS", "5000")),
            max_query_cost=float(e("MAX_QUERY_COST", "1000000")),
            auto_seed_demo=_bool(e("AUTO_SEED_DEMO"), False),
            sample_values=_bool(e("SAMPLE_VALUES"), True),
            llm_provider=(e("LLM_PROVIDER") or cls.llm_provider).strip().lower(),
            llm_model=(e("LLM_MODEL") or cls.llm_model).strip(),
            groq_api_key=e("GROQ_API_KEY") or None,
            max_question_chars=int(e("MAX_QUESTION_CHARS", "500")),
            pii_mode=e("PII_MODE", "mask").lower(),
            max_retries=int(e("MAX_RETRIES", "2")),
            confidence_threshold=float(e("CONFIDENCE_THRESHOLD", "0.6")),
            self_consistency_samples=int(e("SELF_CONSISTENCY_SAMPLES", "0")),
            generate_summary=_bool(e("GENERATE_SUMMARY"), True),
            api_key=e("API_KEY") or None,
        )
