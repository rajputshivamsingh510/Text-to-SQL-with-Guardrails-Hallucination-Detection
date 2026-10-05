"""Evaluation harness. Produces the numbers you put in your README / report.

  python eval/run_eval.py --guardrails-only   # no LLM or API key needed: input guard + SQL guard attack suite
  python eval/run_eval.py                     # full run against the real LLM (needs ANTHROPIC_API_KEY)

Metrics
  execution accuracy   result rows equal the gold query's rows (order- and column-order-insensitive)
  false-block rate     answerable questions that were not answered with status "ok"
  safe-refusal rate    adversarial / unanswerable questions that ended in an expected non-answer status
  attack-SQL block     malicious SQL strings rejected by the SQL guardrail
"""
import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import make_engine, run_select, sqlglot_dialect  # noqa: E402
from app.guardrails.input_guard import check_input  # noqa: E402
from app.guardrails.sql_guard import validate_sql  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.main import build_pipeline  # noqa: E402
from app.schema import load_schema  # noqa: E402



DATA = json.loads((pathlib.Path(__file__).parent / "eval_set.json").read_text())


def _norm(rows):
    def cell(v):
        return str(round(v, 4)) if isinstance(v, float) else str(v).strip().lower()
    return Counter(tuple(sorted(cell(v) for v in row)) for row in rows)


def attack_suite(settings: Settings) -> tuple[int, int]:
    engine = make_engine(settings.database_url, read_only=True)
    schema = load_schema(engine, settings)
    dialect = sqlglot_dialect(engine)
    blocked = [s for s in DATA["attack_sql"] if not validate_sql(s, schema, dialect, settings.max_rows).ok]
    for s in DATA["attack_sql"]:
        if s not in blocked:
            print(f"  !! NOT BLOCKED: {s}")
    questions = [q["question"] for q in DATA["must_not_succeed"] if q["expect"] == ["blocked"]]
    in_blocked = [q for q in questions if not check_input(q).allowed]
    for q in questions:
        if q not in in_blocked:
            print(f"  !! INPUT NOT BLOCKED: {q}")
    print(f"SQL guardrail:   {len(blocked)}/{len(DATA['attack_sql'])} attack queries rejected")
    print(f"Input guardrail: {len(in_blocked)}/{len(questions)} injection/destructive prompts blocked")
    return len(blocked) + len(in_blocked), len(DATA["attack_sql"]) + len(questions)


def full_run(settings: Settings) -> dict:
    llm = build_llm(settings)
    if llm is None:
        sys.exit("Set GROQ_API_KEY for the full evaluation, or use --guardrails-only.")
    pipe = build_pipeline(settings, llm)
    correct, false_blocks, latencies, details = 0, 0, [], []
    for item in DATA["answerable"]:
        resp = pipe.run(item["question"])
        latencies.append(resp.latency_ms)
        gold = run_select(pipe.engine, item["gold_sql"], 10_000)
        match = resp.status == "ok" and _norm(resp.rows) == _norm(gold.rows)
        correct += match
        false_blocks += resp.status != "ok"
        details.append({"question": item["question"], "status": resp.status, "match": match,
                        "attempts": resp.attempts, "confidence": resp.confidence, "sql": resp.sql})
        print(f"  {'PASS' if match else 'FAIL'} [{resp.status:<12}] a={resp.attempts} conf={resp.confidence} :: {item['question']}")
    refused = 0
    for item in DATA["must_not_succeed"]:
        resp = pipe.run(item["question"])
        good = resp.status in item["expect"] and not (resp.status == "ok")
        refused += good
        details.append({"question": item["question"], "status": resp.status, "safe": good})
        print(f"  {'PASS' if good else 'FAIL'} [{resp.status:<12}] :: {item['question']}")
    n_a, n_m = len(DATA["answerable"]), len(DATA["must_not_succeed"])
    report = {
        "model": settings.llm_model,
        "execution_accuracy": round(correct / n_a, 3), "false_block_rate": round(false_blocks / n_a, 3),
        "safe_refusal_rate": round(refused / n_m, 3), "median_latency_ms": int(statistics.median(latencies)),
        "details": details,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    (pathlib.Path(__file__).parent / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails-only", action="store_true")
    args = ap.parse_args()
    settings = Settings.from_env()
    ok, total = attack_suite(settings)
    if not args.guardrails_only:
        full_run(settings)
    sys.exit(0 if ok == total else 1)
