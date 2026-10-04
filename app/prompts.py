"""Prompt templates and parsers for the three LLM roles: SQL writer, reviewer (judge), summariser."""
from __future__ import annotations

import json
import re

SQL_SYSTEM = """You are a careful SQL generator for a {dialect} database. Write ONE read-only SELECT query that answers the user's question.

Rules:
- Use ONLY the tables and columns listed in the schema. Never invent tables, columns or functions.
- Only SELECT (CTEs are fine). Never write INSERT/UPDATE/DELETE/DDL and never write more than one statement.
- Never guess filter values. Text values are case-sensitive and must match the data exactly; when a column lists its values, use one of them.
- Join with the foreign keys shown in the schema and qualify columns with table aliases.
- Revenue / sales amount = quantity * unit_price from order_items unless the schema says otherwise.
- Add ORDER BY and LIMIT for "top", "best", "latest" style questions.
- If the question cannot be answered from this schema, do not guess. Reply with <cannot_answer>one short reason</cannot_answer>.

Output format: the SQL inside <sql></sql> tags and nothing else."""

JUDGE_SYSTEM = """You are a strict reviewer of SQL queries for a data-analytics assistant. You are given a user question, the database schema, the SQL that was run, and a sample of its output.
Decide whether the SQL and its output actually answer the question. Look for: wrong or missing filters, wrong aggregation, wrong join, wrong metric, ignored parts of the question, and columns that do not match what was asked.
Respond with JSON only, no prose, in exactly this shape:
{"answers_question": true|false, "score": <number 0 to 1>, "issues": ["short issue", ...]}"""

SUMMARY_SYSTEM = """You write short answers for a data-analytics assistant.
Answer the question in at most 3 sentences using ONLY the rows provided. Quote numbers exactly as they appear in the rows; do not compute new numbers, estimate, or round. If there are no rows, say that no matching data was found."""


def build_sql_user_prompt(question: str, schema_text: str, examples: list[tuple[str, str]], feedback: str | None) -> str:
    parts = ["Schema:\n" + schema_text]
    if examples:
        parts.append("Example questions with correct SQL:\n" + "\n\n".join(f"Q: {q}\nSQL: {s}" for q, s in examples))
    parts.append(f"Question: {question}")
    if feedback:
        parts.append(
            "Your previous attempt was rejected for this reason:\n"
            f"{feedback}\nWrite a corrected query (or <cannot_answer> if it truly cannot be answered)."
        )
    return "\n\n".join(parts)


def format_rows(columns: list[str], rows: list[list], limit: int = 15) -> str:
    lines = [" | ".join(columns)]
    for r in rows[:limit]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in r))
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


def build_judge_user_prompt(question: str, schema_text: str, sql: str, columns: list[str], rows: list[list], row_count: int) -> str:
    return (
        f"Question: {question}\n\nSchema:\n{schema_text}\n\nSQL that was run:\n{sql}\n\n"
        f"Output ({row_count} rows):\n{format_rows(columns, rows, limit=10)}"
    )


def build_summary_user_prompt(question: str, columns: list[str], rows: list[list], row_count: int, truncated: bool) -> str:
    note = " (result was truncated at the row limit)" if truncated else ""
    return f"Question: {question}\n\nRows ({row_count}{note}):\n{format_rows(columns, rows, limit=25)}"


# --- parsers ---------------------------------------------------------------------------------------------------

_SQL_TAG = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
_CANNOT = re.compile(r"<cannot_answer>(.*?)</cannot_answer>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_sql_response(text: str) -> tuple[str | None, str | None]:
    """Returns (sql, cannot_answer_reason). Tolerates markdown fences if the model ignores the tags."""
    if m := _CANNOT.search(text):
        return None, m.group(1).strip() or "The question cannot be answered from this schema."
    if m := _SQL_TAG.search(text):
        return m.group(1).strip().rstrip(";").strip(), None
    if m := _FENCE.search(text):
        return m.group(1).strip().rstrip(";").strip(), None
    stripped = text.strip().rstrip(";").strip()
    return (stripped or None), None


def parse_json_object(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
