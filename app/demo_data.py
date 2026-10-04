"""Deterministic demo database ("shop"): works on SQLite and Postgres."""
from __future__ import annotations

import datetime as dt
import random

from sqlalchemy import (Column, Date, ForeignKey, Integer, MetaData, Numeric, String, Table, inspect)
from sqlalchemy.engine import Engine

metadata = MetaData()

customers = Table(
    "customers", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), nullable=False),
    Column("email", String(150), nullable=False),  # PII: hidden via BLOCKED_COLUMNS / column-level GRANT
    Column("phone", String(30)),                   # PII
    Column("country", String(50), nullable=False),
    Column("city", String(50)),
    Column("signup_date", Date, nullable=False),
)
products = Table(
    "products", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), nullable=False),
    Column("category", String(50), nullable=False),
    Column("price", Numeric(10, 2), nullable=False),
    Column("stock", Integer, nullable=False),
)
orders = Table(
    "orders", metadata,
    Column("id", Integer, primary_key=True),
    Column("customer_id", Integer, ForeignKey("customers.id"), nullable=False),
    Column("order_date", Date, nullable=False),
    Column("status", String(20), nullable=False),
)
order_items = Table(
    "order_items", metadata,
    Column("id", Integer, primary_key=True),
    Column("order_id", Integer, ForeignKey("orders.id"), nullable=False),
    Column("product_id", Integer, ForeignKey("products.id"), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("unit_price", Numeric(10, 2), nullable=False),
)

_PLACES = [
    ("India", ["Mumbai", "Delhi", "Ludhiana", "Bengaluru"]),
    ("United States", ["New York", "Austin", "Seattle"]),
    ("United Kingdom", ["London", "Manchester"]),
    ("Germany", ["Berlin", "Munich"]),
    ("Canada", ["Toronto", "Vancouver"]),
    ("Australia", ["Sydney", "Melbourne"]),
]
_FIRST = ["Aarav", "Priya", "Liam", "Emma", "Noah", "Olivia", "Rohan", "Sofia", "Karan", "Mia", "Arjun", "Zoe",
          "Ethan", "Ananya", "Lucas", "Isha", "Mason", "Chloe", "Vikram", "Grace"]
_LAST = ["Sharma", "Singh", "Smith", "Johnson", "Brown", "Patel", "Garcia", "Muller", "Wilson", "Taylor", "Gill", "Khan"]
_PRODUCTS = [
    ("Wireless Mouse", "Electronics", 19.99), ("Mechanical Keyboard", "Electronics", 79.50),
    ("USB-C Hub", "Electronics", 34.00), ("Noise-Cancelling Headphones", "Electronics", 149.00),
    ("Webcam HD", "Electronics", 49.99), ("Laptop Stand", "Office", 29.90), ("Desk Lamp", "Office", 24.50),
    ("Notebook Pack", "Office", 9.99), ("Ergonomic Chair", "Office", 199.00), ("Standing Desk", "Office", 349.00),
    ("Yoga Mat", "Fitness", 22.00), ("Dumbbell Set", "Fitness", 89.00), ("Resistance Bands", "Fitness", 14.99),
    ("Water Bottle", "Fitness", 12.50), ("Running Shoes", "Fitness", 95.00), ("Coffee Beans 1kg", "Grocery", 18.75),
    ("Green Tea Box", "Grocery", 8.99), ("Protein Bars 12pk", "Grocery", 21.00), ("Olive Oil 1L", "Grocery", 13.40),
    ("Basmati Rice 5kg", "Grocery", 16.20),
]
_STATUSES = ["completed", "completed", "completed", "shipped", "pending", "cancelled"]


def has_data(engine: Engine) -> bool:
    return "customers" in inspect(engine).get_table_names()


def seed(engine: Engine, *, n_customers: int = 60, n_orders: int = 400, seed_value: int = 42) -> None:
    """Create tables and insert deterministic demo data. Needs a write-capable engine."""
    rnd = random.Random(seed_value)
    metadata.drop_all(engine)
    metadata.create_all(engine)
    today = dt.date(2025, 12, 31)

    cust_rows = []
    for i in range(1, n_customers + 1):
        country, cities = rnd.choice(_PLACES)
        name = f"{rnd.choice(_FIRST)} {rnd.choice(_LAST)}"
        cust_rows.append(dict(
            id=i, name=name, email=f"{name.lower().replace(' ', '.')}{i}@example.com",
            phone=f"+1-555-{rnd.randint(1000, 9999)}", country=country, city=rnd.choice(cities),
            signup_date=today - dt.timedelta(days=rnd.randint(200, 1100)),
        ))
    prod_rows = [dict(id=i, name=n, category=c, price=p, stock=rnd.randint(0, 300))
                 for i, (n, c, p) in enumerate(_PRODUCTS, start=1)]
    order_rows, item_rows, item_id = [], [], 1
    for oid in range(1, n_orders + 1):
        order_rows.append(dict(id=oid, customer_id=rnd.randint(1, n_customers),
                               order_date=today - dt.timedelta(days=rnd.randint(0, 540)),
                               status=rnd.choice(_STATUSES)))
        for _ in range(rnd.randint(1, 4)):
            prod = rnd.choice(prod_rows)
            item_rows.append(dict(id=item_id, order_id=oid, product_id=prod["id"],
                                  quantity=rnd.randint(1, 5), unit_price=prod["price"]))
            item_id += 1

    with engine.begin() as conn:
        conn.execute(customers.insert(), cust_rows)
        conn.execute(products.insert(), prod_rows)
        conn.execute(orders.insert(), order_rows)
        conn.execute(order_items.insert(), item_rows)
