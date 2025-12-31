from contextlib import contextmanager

import psycopg

from .config import DATABASE_URL


def _require_db_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    return DATABASE_URL


@contextmanager
def get_db():
    conn = psycopg.connect(_require_db_url())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_db_cursor():
    with get_db() as conn:
        with conn.cursor() as cursor:
            yield conn, cursor
