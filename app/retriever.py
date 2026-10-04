"""Schema RAG: a small BM25 retriever over table documentation and example question/SQL pairs.

No embeddings API and no heavy dependencies, so it runs comfortably on a small Render instance.
To swap in embeddings (or LlamaIndex / pgvector), implement the same `retrieve()` signature.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from app.schema import SchemaInfo, TableInfo

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be",
    "me", "my", "i", "show", "give", "list", "tell", "what", "which", "who", "how", "many", "much", "do",
    "does", "did", "with", "by", "from", "all", "get", "find", "per", "than", "that", "this", "it", "as",
    "please", "can", "you", "us", "our", "we", "have", "has", "had",
}


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower().replace("_", " ")) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d in self.docs:
            score, dl = 0.0, len(d) or 1
            for t in query:
                if t not in self.idf:
                    continue
                tf = d.count(t)
                if tf:
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(score)
        return out


@dataclass
class Example:
    question: str
    sql: str


@dataclass
class Retrieval:
    tables: list[TableInfo]
    examples: list[Example]
    top_score: float

    @property
    def in_scope(self) -> bool:
        """Zero lexical overlap with every table/column/description means the question is not about this database."""
        return self.top_score > 0


def load_examples(path: Path) -> list[Example]:
    if not path.exists():
        return []
    return [Example(**e) for e in json.loads(path.read_text(encoding="utf-8"))]


class SchemaRetriever:
    def __init__(self, schema: SchemaInfo, examples: list[Example] | None = None,
                 k_tables: int = 4, k_examples: int = 3, max_tables: int = 6):
        self.schema, self.examples = schema, examples or []
        self.k_tables, self.k_examples, self.max_tables = k_tables, k_examples, max_tables
        self._tables = list(schema.tables.values())
        self._table_bm25 = BM25([self._table_doc(t) for t in self._tables])
        self._example_bm25 = BM25([tokenize(e.question) for e in self.examples])

    @staticmethod
    def _table_doc(t: TableInfo) -> list[str]:
        toks = tokenize(t.name) * 3 + tokenize(t.description)
        for c in t.columns:
            toks += tokenize(c.name) + tokenize(c.description)
            toks += [tok for v in c.sample_values for tok in tokenize(v)]
        return toks

    def retrieve(self, question: str) -> Retrieval:
        q = tokenize(question)
        scores = self._table_bm25.scores(q) if self._tables else []
        ranked = sorted(zip(scores, self._tables), key=lambda x: -x[0])
        top = ranked[0][0] if ranked else 0.0
        chosen = [t for s, t in ranked[: self.k_tables] if s > 0] or ([] if top == 0 else [ranked[0][1]])

        # One hop along foreign keys (both directions) so joins like orders -> order_items still work.
        names = {t.name for t in chosen}
        for t in list(chosen):
            neighbours = t.referenced_tables() | {o.name for o in self._tables if t.name in o.referenced_tables()}
            for n in sorted(neighbours):
                if n in self.schema.tables and n not in names and len(names) < self.max_tables:
                    names.add(n)
        tables = [self.schema.tables[n] for n in sorted(names)]

        examples: list[Example] = []
        if self.examples:
            ex_scores = self._example_bm25.scores(q)
            ex_ranked = sorted(zip(ex_scores, self.examples), key=lambda x: -x[0])
            examples = [e for s, e in ex_ranked[: self.k_examples] if s > 0]
        return Retrieval(tables=tables, examples=examples, top_score=top)
