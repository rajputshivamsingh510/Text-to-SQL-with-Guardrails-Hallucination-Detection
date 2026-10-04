"""Schema introspection. This is the single source of truth for what the LLM may see and what queries may touch."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import inspect, types as sa_types
from sqlalchemy.engine import Engine

from app.config import Settings


@dataclass
class ColumnInfo:
    name: str
    type: str  # coarse type used for sqlglot: INT | DECIMAL | TEXT | DATE | TIMESTAMP | BOOLEAN
    raw_type: str = ""
    description: str = ""
    primary_key: bool = False
    references: str | None = None  # "table.column"
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    description: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def referenced_tables(self) -> set[str]:
        return {c.references.split(".")[0] for c in self.columns if c.references}

    def prompt_text(self) -> str:
        lines = [f"Table {self.name}" + (f": {self.description}" if self.description else "")]
        for c in self.columns:
            bits = [f"  - {c.name} {c.raw_type or c.type}"]
            if c.primary_key:
                bits.append("PRIMARY KEY")
            if c.references:
                bits.append(f"FK -> {c.references}")
            if c.description:
                bits.append(f"({c.description})")
            if c.sample_values:
                bits.append("values: " + ", ".join(repr(v) for v in c.sample_values))
            lines.append(" ".join(bits))
        return "\n".join(lines)


@dataclass
class SchemaInfo:
    tables: dict[str, TableInfo]
    dialect: str

    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        return {t.name: {c.name: c.type for c in t.columns} for t in self.tables.values()}

    def table_names(self) -> set[str]:
        return set(self.tables)

    def has_column(self, table: str, column: str) -> bool:
        t = self.tables.get(table.lower())
        return bool(t) and column.lower() in {c.name.lower() for c in t.columns}


def _coarse_type(t) -> str:
    if isinstance(t, sa_types.Boolean):
        return "BOOLEAN"
    if isinstance(t, sa_types.Integer):
        return "INT"
    if isinstance(t, (sa_types.Numeric, sa_types.Float)):
        return "DECIMAL"
    if isinstance(t, sa_types.DateTime):
        return "TIMESTAMP"
    if isinstance(t, sa_types.Date):
        return "DATE"
    return "TEXT"


def load_notes(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_schema(engine: Engine, settings: Settings, notes: dict | None = None) -> SchemaInfo:
    notes = notes if notes is not None else load_notes(settings.data_dir / "schema_notes.json")
    table_notes = notes.get("tables", {})
    insp = inspect(engine)
    blocked = set(settings.blocked_columns)
    allowed = set(settings.allowed_tables)

    tables: dict[str, TableInfo] = {}
    for name in sorted(insp.get_table_names()):
        lname = name.lower()
        if lname.startswith("sqlite_") or (allowed and lname not in allowed):
            continue
        pk = set(insp.get_pk_constraint(name).get("constrained_columns") or [])
        fks = {}
        for fk in insp.get_foreign_keys(name):
            for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fks[col] = f"{fk['referred_table']}.{ref_col}"
        tnote = table_notes.get(lname, {})
        cols = []
        for col in insp.get_columns(name):
            if f"{lname}.{col['name'].lower()}" in blocked:
                continue
            cols.append(
                ColumnInfo(
                    name=col["name"].lower(),
                    type=_coarse_type(col["type"]),
                    raw_type=str(col["type"]),
                    description=tnote.get("columns", {}).get(col["name"].lower(), ""),
                    primary_key=col["name"] in pk,
                    references=fks.get(col["name"]),
                )
            )
        tables[lname] = TableInfo(name=lname, description=tnote.get("description", ""), columns=cols)

    # Drop foreign keys that point at tables we are not exposing.
    for t in tables.values():
        for c in t.columns:
            if c.references and c.references.split(".")[0].lower() not in tables:
                c.references = None

    schema = SchemaInfo(tables=tables, dialect=engine.dialect.name)
    if settings.sample_values:
        _attach_sample_values(engine, schema)
    return schema


def _attach_sample_values(engine: Engine, schema: SchemaInfo, max_distinct: int = 12) -> None:
    """Show the LLM the actual values of low-cardinality text columns so it stops guessing 'USA' vs 'United States'."""
    from sqlalchemy import text

    q = engine.dialect.identifier_preparer.quote
    for table in list(schema.tables.values())[:50]:
        for col in table.columns:
            if col.type != "TEXT" or col.primary_key or col.references:
                continue
            try:
                with engine.connect() as conn:
                    rows = conn.execute(
                        text(f"SELECT DISTINCT {q(col.name)} FROM {q(table.name)} "
                             f"WHERE {q(col.name)} IS NOT NULL LIMIT {max_distinct + 1}")
                    ).fetchall()
            except Exception:  # pragma: no cover - sampling is best effort
                continue
            if 0 < len(rows) <= max_distinct:
                col.sample_values = sorted(str(r[0]) for r in rows)
