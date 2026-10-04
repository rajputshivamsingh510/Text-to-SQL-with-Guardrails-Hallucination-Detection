import json

import pytest

from app.guardrails.sql_guard import validate_sql
from app.retriever import load_examples
from app.config import ROOT

GOOD = [
    "SELECT COUNT(*) FROM orders",
    "SELECT c.country, COUNT(*) AS n FROM customers c GROUP BY c.country ORDER BY n DESC",
    "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.category",
    "WITH big AS (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id) SELECT * FROM big WHERE n > 3",
    "SELECT name FROM customers UNION SELECT name FROM products",
    "SELECT name FROM customers WHERE name LIKE 'a:b%'",
    "SELECT name FROM products WHERE price > (SELECT AVG(price) FROM products)",
]

BAD = {
    "DROP TABLE customers": "not_select",
    "DELETE FROM orders": "not_select",
    "UPDATE orders SET status = 'x'": "not_select",
    "INSERT INTO orders (id) VALUES (1)": "not_select",
    "SELECT 1; DROP TABLE orders": "multiple_statements",
    "SELECT * FROM sqlite_master": "unknown_table",
    "SELECT * FROM pg_catalog.pg_user": "unknown_table",
    "SELECT * FROM information_schema.tables": "unknown_table",
    "SELECT * FROM secrets": "unknown_table",
    "SELECT revenue FROM orders": "unknown_column",
    "SELECT c.loyalty_tier FROM customers c": "unknown_column",
    "SELECT email FROM customers": "unknown_column",          # hidden (blocked) column
    "SELECT phone FROM customers": "unknown_column",
    "SELECT * INTO backup FROM customers": "forbidden_construct",
    "SELECT readfile('/etc/passwd')": "forbidden_function",
    "SELECT randomblob(1000000000)": "forbidden_function",
    "SELECT load_extension('x')": "forbidden_function",
    "PRAGMA table_info(customers)": "not_select",
    "ATTACH DATABASE 'x.db' AS x": "not_select",
    "SELECT FROM WHERE": "parse_error",
    "": "empty",
}


@pytest.mark.parametrize("sql", GOOD)
def test_valid_queries_pass(sql, schema):
    r = validate_sql(sql, schema, "sqlite", 200)
    assert r.ok, r.feedback()
    assert "LIMIT" in r.sql.upper()


@pytest.mark.parametrize("sql,code", list(BAD.items()))
def test_dangerous_or_hallucinated_queries_are_rejected(sql, code, schema):
    r = validate_sql(sql, schema, "sqlite", 200)
    assert not r.ok and code in r.codes, (r.codes, r.feedback())


def test_hallucination_flag_and_helpful_feedback(schema):
    r = validate_sql("SELECT revenue FROM orders", schema, "sqlite")
    assert r.is_hallucination and "do not invent" in r.feedback().lower()
    unknown = validate_sql("SELECT * FROM secrets", schema, "sqlite")
    assert unknown.is_hallucination and "Available tables" in unknown.feedback()


def test_limit_is_enforced_and_capped(schema):
    assert validate_sql("SELECT id FROM orders", schema, "sqlite", 50).sql.endswith("LIMIT 50")
    assert validate_sql("SELECT id FROM orders LIMIT 999999", schema, "sqlite", 50).sql.endswith("LIMIT 50")
    assert validate_sql("SELECT id FROM orders LIMIT 5", schema, "sqlite", 50).sql.endswith("LIMIT 5")


def test_select_star_cannot_leak_hidden_columns(schema):
    r = validate_sql("SELECT * FROM customers", schema, "sqlite")
    assert r.ok and "email" not in r.sql and "phone" not in r.sql and "country" in r.sql
    r2 = validate_sql("SELECT c.* FROM customers c JOIN orders o ON o.customer_id = c.id", schema, "sqlite")
    assert r2.ok and "email" not in r2.sql


@pytest.mark.parametrize("sql,code", [
    ("SELECT pg_sleep(10)", "forbidden_function"),
    ("SELECT pg_read_file('/etc/passwd')", "forbidden_function"),
    ("SELECT * FROM generate_series(1, 1000000000)", "forbidden_function"),
    ("SELECT repeat('a', 1000000000)", "forbidden_function"),
    ("SELECT current_setting('server_version')", "forbidden_function"),
    ("SELECT id FROM orders FOR UPDATE", "forbidden_construct"),
    ("COPY customers TO '/tmp/x'", "not_select"),
    ("SELECT * FROM pg_catalog.pg_shadow", "unknown_table"),
    ("SELECT id FROM public.orders", None),                          # public schema is fine
    ("SELECT dblink('host=x', 'select 1')", "forbidden_function"),
])
def test_postgres_dialect_attacks(sql, code, schema):
    r = validate_sql(sql, schema, "postgres", 200)
    if code is None:
        assert r.ok, r.feedback()
    else:
        assert not r.ok and code in r.codes, (r.codes, r.feedback())


def test_every_example_query_passes_the_guard(schema):
    for ex in load_examples(ROOT / "data" / "examples.json"):
        r = validate_sql(ex.sql, schema, "sqlite")
        assert r.ok, (ex.question, r.feedback())
