# Text-to-SQL with Guardrails & Hallucination Detection

Ask questions about a database in plain English. An LLM writes the SQL, but nothing it produces is trusted:
every query is validated, executed read-only, and fact-checked before an answer is shown.

```
question
  │
  ▼  1. Input guardrail      injection phrases, destructive intent, PII masking, length
  ▼  2. Schema RAG           BM25 over table docs → only the relevant tables reach the LLM  (+ scope check)
  ▼  3. LLM writes SQL       may answer <cannot_answer> instead of guessing
  ▼  4. SQL guardrail        parse tree: one SELECT only, table/column allowlist, no dangerous functions, LIMIT forced
  ▼  5. Execute              read-only role · read-only transaction · statement timeout · cost ceiling
  ▼  6. Hallucination checks value grounding · result sanity · LLM judge · (self-consistency) · answer grounding
  │       └─ fail → retry with the exact problem fed back to the LLM (max 2), else "I'm not sure"
  ▼  7. Grounded answer      numbers in the summary must exist in the rows, otherwise the summary is withheld
response = answer + SQL + rows + confidence + per-check signals + full guardrail trace
```

## What each layer catches

| Layer | Example it stops |
|---|---|
| Input guardrail | "Ignore previous instructions…", `'; DROP TABLE orders; --`, "delete all cancelled orders", emails/phones/Aadhaar/PAN/cards sent to the LLM |
| Scope check | "What's the weather in Delhi?" (no overlap with the schema, so no LLM call is made) |
| SQL guardrail | `DROP/DELETE/UPDATE`, stacked statements, `SELECT … INTO`, `FOR UPDATE`, `pg_sleep`, `pg_read_file`, `generate_series` bombs, `pg_catalog` / `information_schema`, **hallucinated tables and columns**, unbounded result sets, `SELECT *` leaking hidden columns |
| Value grounding | `WHERE country = 'USA'` when the data says `'United States'`: valid SQL, silently wrong answer |
| LLM judge | Right syntax, wrong metric/join/filter for what was asked |
| Self-consistency (optional) | One query disagrees with several independently generated ones |
| Answer grounding | A summary that says "total of 150" when 150 was never in the result |
| Database role | If every check above were bypassed: no writes, no PII columns, 5 s timeout |

## Quickstart (local, no Postgres needed)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                     # put your ANTHROPIC_API_KEY in it
set -a; source .env; set +a                              # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python scripts/seed_demo.py                              # creates demo.db (SQLite) with a small shop dataset
uvicorn app.main:app --reload                            # API on :8000  (docs at /docs)
streamlit run ui/streamlit_app.py                        # UI on :8501   (second terminal)
```

Try the buttons in the sidebar: they include a hallucination bait (`USA` vs `United States`), an unanswerable question
(no loyalty-tier column), a PII request, and a prompt injection.

## Tests and evaluation

```bash
pytest                                      # ~100 tests, no API key needed (scripted fake LLM)
python eval/run_eval.py --guardrails-only   # attack suite: malicious SQL + injection prompts, no LLM needed
python eval/run_eval.py                     # full run with the real LLM → eval/report.json
```
The full run reports execution accuracy, false-block rate and safe-refusal rate: put those numbers in your report.
Postgres integration tests run when `TEST_PG_RO_URL` points at a database prepared with `scripts/create_readonly_role.sql`.

## Configuration

All via environment variables (see `.env.example` for the full list). The important ones:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `LLM_MODEL` | LLM access. Default model is a small, cheap one; use a larger one for accuracy |
| `DATABASE_URL` | What the API queries with. Use the **read-only role** in production |
| `ADMIN_DATABASE_URL` | Seeding only |
| `BLOCKED_COLUMNS` | `table.column` list hidden from the LLM *and* unreachable by queries |
| `ALLOWED_TABLES` | Optional table allowlist |
| `PII_MODE` | `mask` (default), `block`, `off` |
| `MAX_RETRIES`, `CONFIDENCE_THRESHOLD`, `SELF_CONSISTENCY_SAMPLES` | Verification strictness vs. cost |
| `API_KEY` | If set, clients must send `X-API-Key` |

## Deploy on Render

1. Push this folder to a GitHub repo.
2. Render dashboard → **New → Blueprint** → pick the repo. Render reads `render.yaml`, creates the database, the API and
   the UI, and asks for `ANTHROPIC_API_KEY` (paste it there; it never goes in git). The API seeds the demo data on first boot
   (`AUTO_SEED_DEMO=true`).
3. When the API is live, copy its URL (e.g. `https://t2sql-api.onrender.com`) into the UI service's `API_URL` env var and redeploy the UI.
   `API_KEY` is generated once and shared with the UI automatically.
4. **Harden (recommended):** create the least-privilege role, then point the API at it.
   ```bash
   psql "<External Database URL from the Render dashboard>" -v ro_password='a-strong-password' -f scripts/create_readonly_role.sql
   ```
   Set the API's `DATABASE_URL` to the same URL with user `t2sql_readonly` and that password, and change `DATABASE_URL` in
   `render.yaml` to `sync: false` so a later Blueprint sync doesn't overwrite it. Re-run the script if you re-seed.
5. Verify: `curl https://<api>/health`, then use the UI's example buttons.

Notes: free web services sleep when idle (the first request after a pause is slow) and free databases are time-limited, so check
Render's current pricing. Embeddings are not used, so memory stays small.

## Using your own database

Point `DATABASE_URL` at it, list sensitive columns in `BLOCKED_COLUMNS`, describe tables in `data/schema_notes.json`
(descriptions and synonyms noticeably improve retrieval and SQL quality), add 5-20 representative question → SQL pairs to
`data/examples.json` (a test asserts every example passes the SQL guardrail), and write your own `eval/eval_set.json`.

## Project layout

```
app/
  guardrails/input_guard.py   input checks          app/hallucination.py   all verification signals
  guardrails/sql_guard.py     SQL validation        app/pipeline.py        orchestration, retries, trace
  retriever.py                BM25 schema RAG       app/main.py            FastAPI service
  schema.py  db.py            introspection / safe execution
ui/streamlit_app.py           chat UI               scripts/               seed + read-only role SQL
eval/                         eval set + harness    tests/                 pytest suite
```

## Known limitations

- The input guardrail is heuristic (regexes). A cleverly paraphrased jailbreak can get past it; that is why the SQL guardrail and the
  database role exist: the model's output is never trusted, whatever the prompt said.
- Identifiers are normalised to lowercase. Postgres tables/columns created with quoted mixed case are not supported.
- Value grounding verifies text `=` / `IN` filters, not `LIKE` patterns, numbers or dates.
- Confidence is a weighted heuristic over the signals, not a calibrated probability. The LLM judge can itself be wrong.
- Schema sampling runs a `SELECT DISTINCT` per low-cardinality text column at startup; set `SAMPLE_VALUES=false` on very large databases.
- Lexical (BM25) retrieval is fine for small schemas; with hundreds of tables and mismatched vocabulary, swap in embeddings
  (`SchemaRetriever.retrieve()` is the only interface to replace).
- SQLite (local mode) has no statement timeout; Postgres does.
- No rate limiting. Add one before exposing the API publicly.
