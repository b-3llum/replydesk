"""Luminal Cloud client (OpenAI-compatible). Every LLM call in ReplyDesk goes through complete().

Each call is logged to data/llm.log with its token counts, including reasoning_tokens.
"""
import logging
import random
import re
import threading
import time

import requests

import config

TEMPERATURE = 0.7
MAX_TOKENS = 220
RETRIES = 3
RATE_LIMIT_RETRIES = 60  # 429s just mean "wait" (the key is shared), so they get their own, larger budget
MAX_PER_SECOND = 9  # Luminal allows about 10 requests/s per key, then answers 429 retry-after 1s
# DeepSeek-V4.1-Flash thinks by default and spends all 220 tokens reasoning, leaving no reply.
# chat_template_kwargs alone stops working once few-shot turns are in the history; the other two
# switches cover that case, so all three are sent.
THINKING_OFF = {"chat_template_kwargs": {"thinking": False}, "reasoning_effort": "none",
                "thinking": {"type": "disabled"}}

log = logging.getLogger("replydesk.llm")
_local = threading.local()
_rate_lock = threading.Lock()
_next_slot = [0.0]


class LLMError(RuntimeError):
    pass


def _setup_log():
    if log.handlers:
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(config.DATA_DIR / "llm.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


_setup_log()


_reasoning_cache = {}


def logged_reasoning_tokens():
    """Reasoning tokens summed over every logged call except smoke tests (data/llm.log)."""
    path = config.DATA_DIR / "llm.log"
    try:
        stat = path.stat()
    except OSError:
        return 0
    key = (stat.st_mtime_ns, stat.st_size)
    if _reasoning_cache.get("key") != key:
        total = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                match = re.search(r"reasoning_tokens=(\d+)", line)
                if match and "tag=smoke" not in line:
                    total += int(match.group(1))
        _reasoning_cache.update(key=key, total=total)
    return _reasoning_cache["total"]


def model_name():
    return config.require("LUMINAL_MODEL")


def _session():
    if not hasattr(_local, "session"):  # one connection pool per thread
        _local.session = requests.Session()
    return _local.session


def _wait_for_slot():
    """Space request starts MAX_PER_SECOND apart across all threads."""
    with _rate_lock:
        now = time.monotonic()
        slot = max(now, _next_slot[0])
        _next_slot[0] = slot + 1.0 / MAX_PER_SECOND
    time.sleep(slot - now)


def clean(text):
    """One plain paragraph: no <think> block, no wrapping quotes, no line breaks."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    text = " ".join(text.split())
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text


def complete(messages, tag=""):
    """One chat completion -> {text, tokens_in, tokens_out, reasoning_tokens, model}.

    Retries network errors, 429, 5xx and empty replies RETRIES times; token counts add up
    across attempts. Raises LLMError when every attempt fails.
    """
    base = config.require("LUMINAL_BASE_URL").rstrip("/")
    key = config.require("LUMINAL_API_KEY")
    body = {"model": model_name(), "messages": messages, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, **THINKING_OFF}
    used = {"in": 0, "out": 0, "reasoning": 0}
    error = ""
    attempt = waits = 0
    while attempt <= RETRIES:
        attempt += 1
        if attempt > 1:
            time.sleep(min(8, 2 ** (attempt - 2)) + random.random())
        _wait_for_slot()
        started = time.time()
        try:
            resp = _session().post(f"{base}/chat/completions", json=body, timeout=60,
                                   headers={"Authorization": f"Bearer {key}"})
        except requests.exceptions.RequestException as e:
            error = type(e).__name__
            log.warning("call tag=%s attempt=%d error=%s", tag, attempt, error)
            continue
        if resp.status_code == 429 and waits < RATE_LIMIT_RETRIES:
            waits += 1
            attempt -= 1  # not a failure: wait as told and go again
            try:
                delay = float(resp.headers.get("retry-after") or 1)
            except ValueError:
                delay = 1.0
            log.info("call tag=%s rate limited, waiting %.1fs (%d)", tag, delay, waits)
            time.sleep(delay + 2 * random.random())
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            error = f"HTTP {resp.status_code}"
            log.warning("call tag=%s attempt=%d error=%s", tag, attempt, error)
            continue
        if not resp.ok:
            raise LLMError(f"Luminal HTTP {resp.status_code}: {resp.text[:300].replace(key, '***')}")
        data = resp.json()
        usage = data.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        used["in"] += usage.get("prompt_tokens") or 0
        used["out"] += usage.get("completion_tokens") or 0
        used["reasoning"] += reasoning
        choice = (data.get("choices") or [{}])[0]
        text = clean((choice.get("message") or {}).get("content"))
        log.log(logging.WARNING if reasoning else logging.INFO,
                "call tag=%s attempt=%d in=%s out=%s reasoning_tokens=%d finish=%s %.1fs words=%d",
                tag, attempt, usage.get("prompt_tokens"), usage.get("completion_tokens"), reasoning,
                choice.get("finish_reason"), time.time() - started, len(text.split()))
        if text:
            return {"text": text, "tokens_in": used["in"], "tokens_out": used["out"],
                    "reasoning_tokens": used["reasoning"], "model": body["model"]}
        error = f"empty reply (finish={choice.get('finish_reason')}, reasoning_tokens={reasoning})"
    raise LLMError(f"Luminal failed after {RETRIES + 1} attempts: {error}")
