"""SQL guardrail. Validates LLM-generated SQL on the parse tree (never with regexes on raw text).

Layers, in order:
  1. exactly one statement, parseable in the target dialect
  2. root must be a SELECT / set operation; no DML, DDL, SELECT INTO, locking reads, or session commands
  3. no dangerous or resource-bomb functions (pg_sleep, lo_import, load_extension, generate_series, ...)
  4. every table must be in the exposed schema (this also blocks pg_catalog / information_schema / sqlite_master)
  5. every column must resolve against the exposed schema  -> catches hallucinated columns before execution
  6. LIMIT is always enforced; SELECT * is expanded to explicit columns so hidden columns cannot leak
  7. the final SQL is re-parsed and structurally re-checked (fail closed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, SqlglotError, TokenError
from sqlglot.optimizer.qualify import qualify

from app.schema import SchemaInfo

MAX_SQL_CHARS = 4000

_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "AlterColumn", "Command", "Merge",
    "Copy", "TruncateTable", "Set", "Use", "Grant", "Revoke", "Transaction", "Commit", "Rollback", "Into",
    "Lock", "Pragma", "Attach", "Detach", "Analyze", "Kill", "Show", "Describe", "Declare", "Refresh",
]
FORBIDDEN_NODES = tuple(c for c in (getattr(exp, n, None) for n in _FORBIDDEN_NODE_NAMES) if c is not None)

_SELECT_ROOTS = tuple(
    c for c in (exp.Select, getattr(exp, "SetOperation", None), getattr(exp, "Union", None)) if c is not None
)

FORBIDDEN_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "randomblob", "zeroblob", "fts3_tokenizer",
    "dblink", "dblink_exec", "copy", "set_config", "current_setting", "generate_series", "repeat",
    "lpad", "rpad", "sleep", "benchmark", "version", "inet_server_addr", "inet_client_addr",
}
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "txid_", "xpath", "query_to_xml", "ts_debug")

_POSITION_SUFFIX = re.compile(r"\.?\s*Line: \d+, Col: \d+")


@dataclass
class Violation:
    code: str
    message: str


@dataclass
class SQLCheck:
    ok: bool
    sql: str | None = None  # the SQL that is safe to execute (normalised, LIMIT enforced)
    violations: list[Violation] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    qualified: exp.Expression | None = None  # fully qualified tree, used for value grounding

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]

    @property
    def is_hallucination(self) -> bool:
        return any(v.code in {"unknown_table", "unknown_column", "ambiguous_column"} for v in self.violations)

    def feedback(self) -> str:
        return "; ".join(v.message for v in self.violations)


def _func_name(node: exp.Func) -> str:
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return (name or "").lower()


def _structural_violations(tree: exp.Expression) -> list[Violation]:
    out: list[Violation] = []
    if not isinstance(tree, _SELECT_ROOTS):
        out.append(Violation("not_select", f"Only SELECT queries are allowed (got {type(tree).__name__.upper()})."))
    for node in tree.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, FORBIDDEN_NODES):
            out.append(Violation("forbidden_construct", f"Forbidden SQL construct: {type(n).__name__.upper()}."))
            break
    for f in tree.find_all(exp.Func):
        name = _func_name(f)
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            out.append(Violation("forbidden_function", f"Function {name}() is not allowed."))
    return out


def _table_violations(tree: exp.Expression, schema: SchemaInfo) -> tuple[list[Violation], list[str]]:
    out: list[Violation] = []
    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}
    used: list[str] = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            out.append(Violation("forbidden_function", "Table-valued functions are not allowed."))
            continue
        name = t.name.lower()
        if t.catalog or (t.db and t.db.lower() != "public"):
            out.append(Violation("unknown_table", f"Schema-qualified table '{t.db}.{t.name}' is not allowed."))
            continue
        if name in cte_names and not t.db:
            continue
        if name not in schema.tables:
            known = ", ".join(sorted(schema.tables))
            out.append(Violation("unknown_table", f"Table '{t.name}' does not exist. Available tables: {known}."))
        elif name not in used:
            used.append(name)
    return out, used


def _projects_star(tree: exp.Expression) -> bool:
    for sel in tree.find_all(exp.Select):
        for e in sel.expressions:
            if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
                return True
    return False


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(max_rows, copy=False)
    expr = limit.args.get("expression")
    if isinstance(expr, exp.Literal) and expr.is_int and int(expr.this) <= max_rows:
        return tree
    limit.set("expression", exp.Literal.number(max_rows))
    return tree


def validate_sql(sql: str, schema: SchemaInfo, dialect: str, max_rows: int = 200) -> SQLCheck:
    sql = (sql or "").strip()
    if not sql:
        return SQLCheck(False, violations=[Violation("empty", "No SQL was produced.")])
    if len(sql) > MAX_SQL_CHARS:
        return SQLCheck(False, violations=[Violation("too_long", "The SQL is unreasonably long.")])

    # 1. parse, exactly one statement
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as e:
        return SQLCheck(False, violations=[Violation("parse_error", f"SQL could not be parsed: {str(e)[:160]}")])
    if len(statements) != 1:
        return SQLCheck(False, violations=[Violation("multiple_statements", "Exactly one statement is allowed.")])
    tree = statements[0]

    # 2-4. structure, functions, tables
    violations = _structural_violations(tree)
    table_violations, tables = _table_violations(tree, schema)
    violations += table_violations
    if violations:
        return SQLCheck(False, violations=violations, tables=tables)

    # 5. column resolution against the exposed schema
    try:
        qualified = qualify(
            tree.copy(), schema=schema.sqlglot_schema(), dialect=dialect,
            validate_qualify_columns=True, identify=False,
        )
    except OptimizeError as e:
        msg = _POSITION_SUFFIX.sub("", str(e))
        code = "ambiguous_column" if "mbiguous" in msg else "unknown_column"
        hint = ""
        if code == "unknown_column":
            hint = " Use only columns listed in the schema; do not invent columns."
        return SQLCheck(False, violations=[Violation(code, msg + "." + hint)], tables=tables)
    except SqlglotError as e:
        return SQLCheck(False, violations=[Violation("unresolvable", f"Query could not be resolved: {str(e)[:160]}")], tables=tables)

    # 6. star expansion (so hidden columns can't leak) + LIMIT
    final_tree = qualified if _projects_star(tree) else tree.copy()
    final_tree = _enforce_limit(final_tree, max_rows)
    final_sql = final_tree.sql(dialect=dialect, comments=False)

    # 7. fail closed: re-parse what we are about to run and re-check it
    try:
        reparsed = [s for s in sqlglot.parse(final_sql, read=dialect) if s is not None]
    except SqlglotError:
        return SQLCheck(False, violations=[Violation("parse_error", "Normalised SQL failed to re-parse.")])
    if len(reparsed) != 1:
        return SQLCheck(False, violations=[Violation("multiple_statements", "Exactly one statement is allowed.")])
    final_violations = _structural_violations(reparsed[0])
    final_table_violations, _ = _table_violations(reparsed[0], schema)
    if final_violations or final_table_violations:
        return SQLCheck(False, violations=final_violations + final_table_violations, tables=tables)

    return SQLCheck(True, sql=final_sql, tables=tables, qualified=qualified)
