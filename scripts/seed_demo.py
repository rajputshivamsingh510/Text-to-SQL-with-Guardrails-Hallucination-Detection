"""Create the demo tables and load deterministic data.

    ADMIN_DATABASE_URL=postgresql://... python scripts/seed_demo.py

Falls back to DATABASE_URL (SQLite by default). WARNING: drops and recreates the demo tables.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.demo_data import seed  # noqa: E402

if __name__ == "__main__":
    settings = Settings.from_env()
    engine = make_engine(settings.effective_admin_url, read_only=False)
    seed(engine)
    print(f"Seeded demo data into {engine.url.render_as_string(hide_password=True)}")
