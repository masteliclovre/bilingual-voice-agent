import os


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or default


DATABASE_URL = _get_env("DATABASE_URL")

AUTH0_DOMAIN = _get_env("AUTH0_DOMAIN")
AUTH0_AUDIENCE = _get_env("AUTH0_AUDIENCE")
AUTH0_ISSUER = _get_env("AUTH0_ISSUER")

WEBHOOK_SECRET = _get_env("WEBHOOK_SECRET")

DEFAULT_TIMEZONE = _get_env("DEFAULT_TIMEZONE", "UTC")
SLA_TARGET_DEFAULT_SEC = int(_get_env("SLA_TARGET_DEFAULT_SEC", "10"))
