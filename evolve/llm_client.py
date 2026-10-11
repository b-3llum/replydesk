"""OpenAI-compatible chat client for Luminal, with thinking forced off.

Two flags are sent. chat_template_kwargs={"thinking": False} was the documented one, but
this endpoint ignores it (probe 2026-10-11: 60/60 completion tokens went to reasoning,
content ''). The top-level "thinking": {"type": "disabled"} is what actually works.
Every call's reasoning_tokens is logged to evolve/logs/llm_calls.jsonl, and a call
that reasons anyway raises, so a silent all-reasoning empty reply can't be scored.
"""
import json
import random
import threading
import time

import requests

from evolve.common import LOG_DIR, config

BASE_URL = config.require("LUMINAL_BASE_URL").rstrip("/")
API_KEY = config.require("LUMINAL_API_KEY")
MODEL = config.require("LUMINAL_MODEL")

# Caps in-flight requests across all threads: run.py nests agent threads around
# score_prompt's own pool, which would otherwise multiply into dozens of requests.
MAX_CONCURRENT = int(config.get("EVOLVE_LLM_CONCURRENCY") or 8)
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
# Luminal rate-limits requests per second, not concurrency (probes 2026-10-11: bursts of ~20
# pass, sequential 1.15 req/s ran clean, paced 2 req/s needed 352 retries for 324 calls).
# Space request starts evenly; retries go through the same pacing.
REQUESTS_PER_SECOND = float(config.get("EVOLVE_LLM_RPS") or 1.1)
_pace_lock = threading.Lock()
_next_start = 0.0


def _pace():
    global _next_start
    with _pace_lock:
        now = time.monotonic()
        start = max(now, _next_start)
        _next_start = start + 1.0 / REQUESTS_PER_SECOND
    time.sleep(max(0.0, start - now))
_session = threading.local()
_log_lock = threading.Lock()
_totals_lock = threading.Lock()
TOTALS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "retries": 0}


class ReasoningLeak(RuntimeError):
    pass


class Aborted(RuntimeError):
    pass


# Set by run.py at its --stop-at deadline: every call (and retry) after that raises Aborted,
# so in-flight candidates fail fast instead of finishing a round past the deadline.
ABORT = threading.Event()


def _http():
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
        _session.s.headers["Authorization"] = f"Bearer {API_KEY}"
    return _session.s


def _retry_after(response):
    try:
        return max(0.5, float(response.headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return None


def chat(messages, temperature=0.0, max_tokens=400, tag="", retries=40):
    """Returns (content, usage). Retries 429/5xx/timeouts with backoff up to 30 s (~20 min total).

    The rate limit is per API key ("x-reject-reason: client-request-rate-limit"), so other
    processes on the same key (e.g. `run.py generate --all`) can hold it for minutes. Retry-After
    says 1 s even then, so it is only a floor under the backoff, never the whole wait.
    """
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
            "thinking": {"type": "disabled"}}
    delay = 2.0
    for attempt in range(retries):
        wait = None
        if ABORT.is_set():
            raise Aborted(f"{tag}: deadline reached")
        try:
            _pace()
            with _slots:
                r = _http().post(f"{BASE_URL}/chat/completions", json=body, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                wait = _retry_after(r)
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            with _totals_lock:
                TOTALS["retries"] += 1
            # sleep outside the slot so waiting threads don't hold concurrency
            time.sleep(max(wait or 0, delay) + random.uniform(0, 1))
            delay = min(delay * 2, 30)
    usage = data.get("usage") or {}
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    content = (data["choices"][0]["message"].get("content") or "").strip()
    _record(tag, usage, reasoning, data["choices"][0].get("finish_reason"))
    if reasoning:
        raise ReasoningLeak(f"{tag}: {reasoning} reasoning tokens despite thinking off")
    return content, usage


def _record(tag, usage, reasoning, finish):
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
           "prompt_tokens": usage.get("prompt_tokens", 0),
           "completion_tokens": usage.get("completion_tokens", 0),
           "reasoning_tokens": reasoning, "finish": finish}
    with _totals_lock:
        TOTALS["calls"] += 1
        TOTALS["prompt_tokens"] += row["prompt_tokens"]
        TOTALS["completion_tokens"] += row["completion_tokens"]
        TOTALS["reasoning_tokens"] += reasoning
    with _log_lock, open(LOG_DIR / "llm_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
