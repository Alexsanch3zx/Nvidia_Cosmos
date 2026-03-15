import os

# Optional: load .env so DATABASE_URL is set (pip install python-dotenv)
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """Return a new connection to the database. Caller must close it."""
    import psycopg2  # type: ignore[import-untyped]
    return psycopg2.connect(DATABASE_URL)
