"""API request/response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "uncertain", "blocked", "out_of_scope", "unanswerable", "rejected", "error"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class TraceStep(BaseModel):
    name: str
    status: Literal["pass", "fail", "warn", "info", "skip"]
    detail: str = ""


class SignalOut(BaseModel):
    name: str
    score: float | None = None
    passed: bool | None = None
    detail: str = ""


class QueryResponse(BaseModel):
    status: Status
    message: str | None = None
    question: str = ""
    answer: str | None = None
    sql: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False
    confidence: float | None = None
    confidence_label: str | None = None
    signals: list[SignalOut] = []
    trace: list[TraceStep] = []
    attempts: int = 0
    warnings: list[str] = []
    latency_ms: int = 0
