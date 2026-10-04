"""The request pipeline. One call to Pipeline.run() walks every guardrail in order and returns a full audit trace.

input guardrail -> schema RAG (+ scope check) -> LLM writes SQL -> SQL guardrail -> read-only execution
-> hallucination checks -> (retry with feedback, up to max_retries) -> grounded summary -> response
"""
from __future__ import annotations

import json
import logging
import time

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app import hallucination as hal
from app.config import Settings
from app.db import QueryResult, estimate_cost, run_select, sqlglot_dialect
from app.guardrails.input_guard import check_input
from app.guardrails.sql_guard import SQLCheck, validate_sql
from app.llm import LLM
from app.models import QueryResponse, SignalOut, TraceStep
from app.prompts import (SQL_SYSTEM, SUMMARY_SYSTEM, build_sql_user_prompt, build_summary_user_prompt,
                         parse_sql_response)
from app.retriever import SchemaRetriever
from app.schema import SchemaInfo

log = logging.getLogger("text2sql")


def _short(e: Exception, n: int = 200) -> str:
    msg = str(getattr(e, "orig", e)).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg[:n]


def _fallback_answer(result: QueryResult) -> str:
    if result.row_count == 0:
        return "No matching data was found."
    if result.row_count == 1 and len(result.columns) == 1:
        return f"{result.columns[0]}: {result.rows[0][0]}"
    return f"The query returned {result.row_count} row(s). See the table below."


class Pipeline:
    def __init__(self, settings: Settings, engine: Engine, schema: SchemaInfo, retriever: SchemaRetriever, llm: LLM):
        self.s, self.engine, self.schema, self.retriever, self.llm = settings, engine, schema, retriever, llm
        self.dialect = sqlglot_dialect(engine)

    # ------------------------------------------------------------------------------------------------------------
    def run(self, question: str) -> QueryResponse:
        t0 = time.time()
        trace: list[TraceStep] = []
        resp = self._run(question, trace)
        resp.trace = trace
        resp.latency_ms = int((time.time() - t0) * 1000)
        log.info(json.dumps({
            "event": "query", "status": resp.status, "confidence": resp.confidence, "attempts": resp.attempts,
            "latency_ms": resp.latency_ms, "question": resp.question, "sql": resp.sql,
        }))
        return resp

    # ------------------------------------------------------------------------------------------------------------
    def _run(self, raw_question: str, trace: list[TraceStep]) -> QueryResponse:
        def step(name: str, status: str, detail: str = "") -> None:
            trace.append(TraceStep(name=name, status=status, detail=detail))

        # 1. input guardrail
        ic = check_input(raw_question, max_chars=self.s.max_question_chars, pii_mode=self.s.pii_mode)
        if not ic.allowed:
            step("input_guardrail", "fail", f"{ic.category}: {ic.reason}")
            return QueryResponse(status="blocked", message=ic.reason, question=ic.sanitized)
        question = ic.sanitized
        warnings: list[str] = []
        if ic.pii_found:
            warnings.append(f"Personal data ({', '.join(ic.pii_found)}) was masked before processing.")
            step("input_guardrail", "warn", f"PII masked: {', '.join(ic.pii_found)}")
        else:
            step("input_guardrail", "pass", "No injection, write intent or PII detected.")

        # 2. schema RAG + scope check
        retrieval = self.retriever.retrieve(question)
        if not retrieval.in_scope:
            step("scope_check", "fail", "The question has no overlap with this database's tables or columns.")
            return QueryResponse(
                status="out_of_scope", question=question, warnings=warnings,
                message="That doesn't look like a question about this database. Try asking about "
                        + ", ".join(sorted(self.schema.tables)) + ".")
        names = ", ".join(t.name for t in retrieval.tables)
        step("schema_retrieval", "pass", f"Selected {len(retrieval.tables)} table(s): {names}; {len(retrieval.examples)} example(s).")
        schema_text = "\n\n".join(t.prompt_text() for t in retrieval.tables)
        examples = [(e.question, e.sql) for e in retrieval.examples]
        system = SQL_SYSTEM.format(dialect=self.dialect)

        # 3. generate -> validate -> execute -> verify, with feedback-driven retries
        feedback: str | None = None
        executed: tuple[SQLCheck, QueryResult, hal.HallucinationReport] | None = None
        last_violation = ""
        attempts = 0
        for attempts in range(1, self.s.max_retries + 2):
            label = f"attempt {attempts}"
            try:
                raw = self.llm.complete(system, build_sql_user_prompt(question, schema_text, examples, feedback))
            except Exception as e:
                log.exception("LLM call failed")
                step("sql_generation", "fail", f"{label}: language model error ({type(e).__name__})")
                return QueryResponse(status="error", question=question, attempts=attempts, warnings=warnings,
                                     message="The language model is unavailable right now. Please try again.")
            sql, cannot = parse_sql_response(raw)
            if cannot:
                step("sql_generation", "info", f"{label}: model declined to guess: {cannot}")
                return QueryResponse(status="unanswerable", question=question, attempts=attempts, warnings=warnings,
                                     message=f"I can't answer that from this database: {cannot}")
            step("sql_generation", "pass", f"{label}: generated SQL")

            guard = validate_sql(sql or "", self.schema, self.dialect, self.s.max_rows)
            if not guard.ok:
                kind = "hallucinated identifier" if guard.is_hallucination else "policy violation"
                step("sql_guardrail", "fail", f"{label}: {kind}: {guard.feedback()}")
                last_violation, feedback = guard.feedback(), guard.feedback()
                continue
            step("sql_guardrail", "pass", f"{label}: SELECT-only, tables/columns verified, LIMIT {self.s.max_rows} enforced.")

            try:
                cost = estimate_cost(self.engine, guard.sql)
                if cost is not None and cost > self.s.max_query_cost:
                    feedback = f"The query is too expensive (planner cost {cost:,.0f}). Add filters or aggregate earlier."
                    step("execution", "fail", f"{label}: {feedback}")
                    continue
                result = run_select(self.engine, guard.sql, self.s.max_rows)
            except SQLAlchemyError as e:
                feedback = f"The database rejected the query: {_short(e)}"
                step("execution", "fail", f"{label}: {feedback}")
                continue
            step("execution", "pass", f"{label}: {result.row_count} row(s) in a read-only transaction.")

            report = self._verify(question, schema_text, system, guard, result)
            for s in report.signals:
                step(f"check:{s.name}", "pass" if s.passed in (True, None) and not s.hard_fail else "fail",
                     f"{label}: {s.detail}")
            executed = (guard, result, report)
            if report.passed:
                break
            feedback = " ".join(report.issues) or "The result did not appear to answer the question."

        # 4. decide the outcome
        if executed is None:
            return QueryResponse(
                status="rejected", question=question, attempts=attempts, warnings=warnings,
                message="I couldn't produce a safe, valid query for that question. "
                        + (f"Last problem: {last_violation}" if last_violation else ""))

        guard, result, report = executed
        ok = report.passed
        answer = None
        if ok:
            answer = self._summarise(question, result, report, warnings, step)
        else:
            warnings.append("The result did not pass all verification checks; treat it as unverified.")
        return QueryResponse(
            status="ok" if ok else "uncertain",
            message=None if ok else "I'm not confident this query answers your question. Review the SQL and the checks below.",
            question=question, answer=answer, sql=guard.sql, columns=result.columns, rows=result.rows,
            row_count=result.row_count, truncated=result.truncated, confidence=report.confidence,
            confidence_label=report.label,
            signals=[SignalOut(name=s.name, score=s.score, passed=s.passed, detail=s.detail) for s in report.signals],
            attempts=attempts, warnings=warnings)

    # ------------------------------------------------------------------------------------------------------------
    def _verify(self, question: str, schema_text: str, system: str, guard: SQLCheck, result: QueryResult) -> hal.HallucinationReport:
        signals = [
            hal.check_value_grounding(guard.qualified, self.engine, self.schema),
            hal.check_result_sanity(result),
        ]
        if not any(s.hard_fail for s in signals):  # don't spend LLM calls on a query we are about to reject
            signals.append(hal.judge_semantics(self.llm, question, schema_text, guard.sql or "", result))
            if self.s.self_consistency_samples > 0:
                signals.append(self._consistency(question, schema_text, system, result))
        return hal.combine(signals, self.s.confidence_threshold)

    def _consistency(self, question: str, schema_text: str, system: str, main: QueryResult) -> hal.Signal:
        alternatives: list[QueryResult | None] = []
        for _ in range(self.s.self_consistency_samples):
            try:
                raw = self.llm.complete(system, build_sql_user_prompt(question, schema_text, [], None), temperature=0.7)
                sql, cannot = parse_sql_response(raw)
                check = validate_sql(sql or "", self.schema, self.dialect, self.s.max_rows) if not cannot else None
                alternatives.append(run_select(self.engine, check.sql, self.s.max_rows) if check and check.ok else None)
            except Exception:
                alternatives.append(None)
        return hal.check_self_consistency(main, alternatives)

    def _summarise(self, question: str, result: QueryResult, report: hal.HallucinationReport,
                   warnings: list[str], step) -> str:
        if not self.s.generate_summary:
            return _fallback_answer(result)
        try:
            text = self.llm.complete(
                SUMMARY_SYSTEM,
                build_summary_user_prompt(question, result.columns, result.rows, result.row_count, result.truncated),
                max_tokens=300).strip()
        except Exception as e:
            step("answer_grounding", "warn", f"Summary unavailable ({type(e).__name__}); showing a plain result description.")
            return _fallback_answer(result)
        sig = hal.check_answer_grounding(text, question, result)
        if sig.passed:
            step("answer_grounding", "pass", sig.detail)
            report.signals.append(sig)
            merged = hal.combine(report.signals, self.s.confidence_threshold)
            report.confidence, report.label = merged.confidence, merged.label
            return text
        # The summary invented numbers: withhold it and show a deterministic description instead.
        step("answer_grounding", "fail", sig.detail + " Summary withheld.")
        report.signals.append(hal.Signal("answer_grounding", None, False, sig.detail + " The generated summary was withheld."))
        warnings.append("The generated summary contained figures not present in the data and was withheld.")
        return _fallback_answer(result)
