"""Paths and the Mongo handle shared by the evolve modules."""
import sys
import threading
from pathlib import Path

EVOLVE_DIR = Path(__file__).resolve().parent
ROOT = EVOLVE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import certifi  # noqa: E402
from pymongo import MongoClient  # noqa: E402

import config  # noqa: E402
from store import _escape_credentials  # noqa: E402  (read-only reuse; percent-encodes raw passwords)

PROMPTS_DIR = EVOLVE_DIR / "prompts"
CACHE_DIR = EVOLVE_DIR / "cache"
LOG_DIR = EVOLVE_DIR / "logs"
SPLIT_PATH = EVOLVE_DIR / "split.json"
for d in (PROMPTS_DIR, CACHE_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

_db = None
_db_lock = threading.Lock()


def db():
    """The replydesk Mongo database (collections: reviews, businesses, prompt_runs)."""
    global _db
    with _db_lock:
        if _db is None:
            client = MongoClient(_escape_credentials(config.require("MONGODB_URI")),
                                 serverSelectionTimeoutMS=8000, tlsCAFile=certifi.where())
            _db = client["replydesk"]
        return _db


def has_text(review):
    return bool((review.get("text") or "").strip())
