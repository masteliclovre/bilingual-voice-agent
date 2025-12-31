import json
from dataclasses import dataclass
from functools import lru_cache

import requests
from jose import jwt
from jose.exceptions import JWTError

from .config import AUTH0_AUDIENCE, AUTH0_DOMAIN, AUTH0_ISSUER


@dataclass
class AuthContext:
    user_id: str
    email: str
    token: str


def _jwks_url() -> str:
    if not AUTH0_DOMAIN:
        raise RuntimeError("AUTH0_DOMAIN is not set.")
    return f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"


@lru_cache(maxsize=1)
def _load_jwks() -> dict:
    response = requests.get(_jwks_url(), timeout=10)
    response.raise_for_status()
    return response.json()


def _get_rsa_key(token: str) -> dict | None:
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    if not kid:
        return None
    jwks = _load_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


def verify_token(token: str) -> AuthContext:
    if not AUTH0_AUDIENCE or not AUTH0_ISSUER:
        raise RuntimeError("AUTH0_AUDIENCE or AUTH0_ISSUER is not set.")

    rsa_key = _get_rsa_key(token)
    if not rsa_key:
        raise JWTError("Unable to find matching RSA key.")

    payload = jwt.decode(
        token,
        rsa_key,
        algorithms=["RS256"],
        audience=AUTH0_AUDIENCE,
        issuer=AUTH0_ISSUER,
    )
    email = payload.get("email") or payload.get("https://example.com/email")
    if not email:
        raise JWTError("Missing email in token.")
    user_id = payload.get("sub", "")
    return AuthContext(user_id=user_id, email=email, token=token)
