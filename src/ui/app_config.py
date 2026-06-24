import os
import secrets

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8501"))
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")

# ---------------------------------------------------------------------------
# Request protection
# ---------------------------------------------------------------------------

MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))  # 50 MB

# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(_PROJECT_ROOT, "data", "uploads"))

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _parse_api_keys(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    return {key.strip() for key in raw.split(",") if key.strip()}


API_KEYS: set[str] = _parse_api_keys(os.getenv("API_KEYS", ""))
