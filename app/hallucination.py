"""Hallucination detection for text-to-SQL.

Identifier hallucinations (made-up tables/columns) are already stopped by the SQL guardrail. This module catches
the subtler failures that still produce valid, runnable SQL:

  value_grounding   filter literals (country = 'USA') that do not exist in the column
  result_sanity     empty or truncated results
  semantic_judge    an LLM reviewer checks that the SQL and output actually answer the question
  self_consistency  (optional) independently sampled SQL variants must return the same rows
  answer_grounding  every number in the natural-language answer must come from the returned rows

Each check returns a Signal. Signals are combined into one confidence score; any "hard fail" forces a retry.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine
from sqlglot import exp

from app.db import QueryResult, value_exists
from app.llm import LLM
from app.prompts import JUDGE_SYSTEM, build_judge_user_prompt, parse_json_object
from app.schema import SchemaInfo

WEIGHTS = {
    "value_grounding": 0.25,
    "semantic_judge": 0.35,
    "result_sanity": 0.15,
    "self_consistency": 0.25,
    "answer_grounding": 0.20,
}


@dataclass
class Signal:
    name: str
    score: float | None  # 0..1; None = not applicable / unavailable (excluded from confidence)
    passed: bool | None
    detail: str
    hard_fail: bool = False


@dataclass
class HallucinationReport:
    confidence: float
    label: str  # high | medium | low
    passed: bool
    signals: list[Signal] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def label_for(confidence: float) -> str:
    return "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"


def combine(signals: list[Signal], threshold: float) -> HallucinationReport:
    scored = [s for s in signals if s.score is not None]
    total_w = sum(WEIGHTS.get(s.name, 0.1) for s in scored)
    confidence = (sum(WEIGHTS.get(s.name, 0.1) * s.score for s in scored) / total_w) if total_w else 0.5
    confidence = round(confidence, 2)
    hard = [s for s in signals if s.hard_fail]
    issues = [s.detail for s in signals if s.hard_fail or (s.score is not None and s.score < 0.7)]
    return HallucinationReport(
        confidence=confidence, label=label_for(confidence),
        passed=(not hard and confidence >= threshold), signals=signals, issues=issues,
    )


# --- 1. value grounding ---------------------------------------------------------------------------------------

def _string_literals(node: exp.Expression) -> list[str]:
    return [node.this] if isinstance(node, exp.Literal) and node.is_string else []


def _column(schema: SchemaInfo, table: str, name: str):
    return next((c for c in schema.tables[table].columns if c.name == name.lower()), None)


def check_value_grounding(qualified: exp.Expression | None, engine: Engine, schema: SchemaInfo,
                          max_checks: int = 8) -> Signal:
    if qualified is None:
        return Signal("value_grounding", None, None, "No parse tree available.")
    alias_to_table = {
        t.alias_or_name.lower(): t.name.lower()
        for t in qualified.find_all(exp.Table)
        if isinstance(t.this, exp.Identifier) and t.name.lower() in schema.tables
    }
    pending: list[tuple[str, str, str]] = []

    def add(col: exp.Column, literal: exp.Expression) -> None:
        table = alias_to_table.get((col.table or "").lower())
        if not table or not schema.has_column(table, col.name):
            return
        column = _column(schema, table, col.name)
        if column is None or column.type != "TEXT":
            return
        for value in _string_literals(literal):
            pending.append((table, column.name, value))

    for node in qualified.find_all(exp.EQ, exp.NEQ):
        a, b = node.this, node.expression
        if isinstance(a, exp.Column):
            add(a, b)
        elif isinstance(b, exp.Column):
            add(b, a)
    for node in qualified.find_all(exp.In):
        if isinstance(node.this, exp.Column):
            for lit in node.expressions:
                add(node.this, lit)

    unique = list(dict.fromkeys(pending))[:max_checks]
    if not unique:
        return Signal("value_grounding", None, True, "No text filter values to verify.")

    missing: list[str] = []
    for table, column, value in unique:
        if value_exists(engine, table, column, value):
            continue
        known = _column(schema, table, column).sample_values
        hint = f" Existing values include: {', '.join(repr(v) for v in known)}." if known else ""
        if value_exists(engine, table, column, value, case_insensitive=True):
            missing.append(f"'{value}' matches {table}.{column} only when case is ignored; use the exact stored casing.{hint}")
        else:
            missing.append(f"The value '{value}' does not exist in {table}.{column}.{hint}")
    score = 1 - len(missing) / len(unique)
    if missing:
        return Signal("value_grounding", round(score, 2), False, " ".join(missing), hard_fail=True)
    return Signal("value_grounding", 1.0, True, f"All {len(unique)} filter value(s) exist in the data.")


# --- 2. result sanity -----------------------------------------------------------------------------------------

def check_result_sanity(result: QueryResult) -> Signal:
    if result.row_count == 0:
        return Signal("result_sanity", 0.5, True,
                      "The query returned no rows; a filter may be wrong, or the data may genuinely be empty.")
    if result.truncated:
        return Signal("result_sanity", 0.9, True, "Result was truncated at the row limit.")
    return Signal("result_sanity", 1.0, True, f"Query returned {result.row_count} row(s).")


# --- 3. semantic judge ----------------------------------------------------------------------------------------

def judge_semantics(llm: LLM, question: str, schema_text: str, sql: str, result: QueryResult) -> Signal:
    try:
        raw = llm.complete(JUDGE_SYSTEM, build_judge_user_prompt(
            question, schema_text, sql, result.columns, result.rows, result.row_count), max_tokens=400)
    except Exception as e:  # reviewer failing must not take the whole request down
        return Signal("semantic_judge", None, None, f"Reviewer unavailable ({type(e).__name__}).")
    obj = parse_json_object(raw)
    if not obj:
        return Signal("semantic_judge", None, None, "Reviewer returned unreadable output; skipped.")
    try:
        score = max(0.0, min(1.0, float(obj.get("score", 1.0 if obj.get("answers_question") else 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    answers = bool(obj.get("answers_question", score >= 0.5))
    issues = [str(i) for i in (obj.get("issues") or [])][:4]
    detail = "; ".join(issues) if issues else "Reviewer found no issues."
    return Signal("semantic_judge", round(score, 2), answers, detail, hard_fail=(not answers and score < 0.5))


# --- 4. self-consistency (optional) ---------------------------------------------------------------------------

def _normalise_rows(result: QueryResult) -> Counter:
    def norm(v):
        return round(v, 4) if isinstance(v, float) else (v.strip().lower() if isinstance(v, str) else v)

    return Counter(tuple(norm(v) for v in row) for row in result.rows)


def check_self_consistency(main: QueryResult, alternatives: list[QueryResult | None]) -> Signal:
    valid = [a for a in alternatives if a is not None]
    if not valid:
        return Signal("self_consistency", None, None, "No alternative queries could be executed.")
    base = _normalise_rows(main)
    alt_counts = [_normalise_rows(a) for a in valid]
    agree = sum(1 for c in alt_counts if c == base)
    score = agree / len(valid)
    detail = f"{agree} of {len(valid)} independently generated queries returned the same rows."
    # Hard fail only when the alternatives agree with each other but not with the main query.
    majority_elsewhere = any(
        c != base and sum(1 for o in alt_counts if o == c) >= 2 for c in alt_counts
    )
    return Signal("self_consistency", round(score, 2), score >= 0.5, detail,
                  hard_fail=(agree == 0 and majority_elsewhere))


# --- 5. answer grounding --------------------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def check_answer_grounding(answer: str, question: str, result: QueryResult) -> Signal:
    allowed: list[float] = [float(result.row_count)] + _numbers(question)
    for row in result.rows:
        for v in row:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                allowed.append(float(v))
            elif isinstance(v, str):
                allowed += _numbers(v)
    claimed = _numbers(answer)
    if not claimed:
        return Signal("answer_grounding", 1.0, True, "The answer contains no figures to verify.")
    ungrounded = [x for x in claimed if not any(abs(x - n) <= max(0.01, 0.005 * abs(n)) for n in allowed)]
    score = 1 - len(ungrounded) / len(claimed)
    if ungrounded:
        shown = ", ".join(f"{x:g}" for x in ungrounded[:5])
        return Signal("answer_grounding", round(score, 2), False, f"Figures not found in the result rows: {shown}.")
    return Signal("answer_grounding", 1.0, True, f"All {len(claimed)} figure(s) in the answer appear in the result rows.")
