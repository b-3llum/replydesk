"""Data for the admin page's "Prompt evolution" card.

    import evolution
    evo = evolution.load()   # a dict, or None until evolve/results.json is written and complete

Reads evolve/ and never writes there (another process owns it):
    evolve/results.json                written once by evolve/run.py final_eval, after the last round
    evolve/runs/<run_id>/prompts/      every prompt that run scored, as <prompt_id>.txt (v0.txt too)
    evolve/prompts/v0.txt, best.txt    the starting prompt and the latest winner
    evolve/cache/generations.jsonl     {"k": sha256(model, prompt, review_id, temperature), "v": reply}
    evolve/split.json                  train / holdout review ids and businesses

The before/after sample is saved in data/evolution_sample.json, keyed by a hash of the best
prompt, so it is looked up (or generated with llm.complete) only once per winning prompt.
"""
import hashlib
import html
import json
import re
import threading

import config
import llm
import store

EVOLVE_DIR = config.BASE_DIR / "evolve"
RESULTS_PATH = EVOLVE_DIR / "results.json"
SPLIT_PATH = EVOLVE_DIR / "split.json"
GENERATIONS_PATH = EVOLVE_DIR / "cache" / "generations.jsonl"
SAMPLE_PATH = config.DATA_DIR / "evolution_sample.json"
GEN_TEMPERATURE = 0.3  # evolve/score.py GEN_TEMPERATURE; part of its generation cache key
SAFE_NAME = re.compile(r"[\w-]+")  # run ids and prompt ids become file names

INK, DOT, MUTED, GRID = "#2F2F2F", "#8FE388", "#6B6F76", "#E6E4DE"

_lock = threading.Lock()
_memo = {"stamp": None, "value": None}


def load():
    """The card data, or None when results.json is missing, unreadable or half-written.

    Cached by the file's mtime and size, so a repeat call costs one stat(). Never raises.
    """
    try:
        stat = RESULTS_PATH.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _lock:
        if _memo["stamp"] != stamp:
            results = _read_json(RESULTS_PATH)
            if not isinstance(results, dict):
                return None  # not memoized: a half-written file is read again on the next call
            try:
                value = _summarize(results)
            except Exception as e:
                print(f"warning: evolution summary failed: {type(e).__name__}")
                return None
            _memo["stamp"], _memo["value"] = stamp, value
        return _memo["value"]


def cached():
    """True when load() would answer from memory (no store lookups or LLM calls)."""
    try:
        stat = RESULTS_PATH.stat()
    except OSError:
        return False
    return _memo["stamp"] == (stat.st_mtime_ns, stat.st_size)


def _summarize(results):
    rounds = _rounds(results)
    baseline = _number(results, "baseline_holdout", "baseline_heldout", "holdout_baseline", "v0_holdout")
    best = _number(results, "best_holdout", "best_heldout", "holdout_best")
    improvement = None
    if baseline and best is not None:
        improvement = (best - baseline) / baseline * 100
    v0_prompt = _v0_prompt(results)
    best_prompt = _best_prompt(results, v0_prompt)
    return {
        "run_id": str(results.get("run_id") or ""),
        "rounds": rounds,
        "baseline": baseline,
        "best": best,
        "improvement_pct": improvement,
        "v0_prompt": v0_prompt,
        "best_prompt": best_prompt,
        "svg": chart_svg(rounds),
        "sample": _sample(results, v0_prompt, best_prompt),
    }


def _number(row, *keys):
    """The value of the first key that holds a real number, as a float; None if none does."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _rounds(results):
    """[(round_number, best_train_score)], starting with v0 as round 0.

    run.py writes "rounds": [{"round": 1, "best_train": 94.46, "best_id": ..., ...}, ...] and keeps
    v0's train score in "baseline_train" (round 0 is not in the list).
    """
    points = {}
    for i, row in enumerate(results.get("rounds") or [], start=1):
        if isinstance(row, dict):
            number = row.get("round", i)
            score = _number(row, "best_train", "best_train_score", "best_score", "score")
        else:
            number, score = i, _number({"score": row}, "score")
        if score is not None and isinstance(number, int):
            points[number] = score
    baseline_train = _number(results, "baseline_train", "v0_train")
    if baseline_train is not None:
        points.setdefault(0, baseline_train)
    return sorted(points.items())


# ---------------------------------------------------------------- prompt texts

def _read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def _run_prompt(results, prompt_id):
    """evolve/runs/<run_id>/prompts/<prompt_id>.txt: the exact text that run scored."""
    run_id = str(results.get("run_id") or "")
    prompt_id = str(prompt_id or "")
    if not SAFE_NAME.fullmatch(run_id) or not SAFE_NAME.fullmatch(prompt_id):
        return ""
    return _read_text(EVOLVE_DIR / "runs" / run_id / "prompts" / f"{prompt_id}.txt")


def _v0_prompt(results):
    return _run_prompt(results, "v0") or _read_text(EVOLVE_DIR / "prompts" / "v0.txt")


def _best_prompt(results, v0_prompt):
    best_id = str(results.get("best_prompt_id") or results.get("best_id") or "")
    if best_id == "v0":
        return v0_prompt
    for key in ("best_prompt", "best_prompt_text"):
        if isinstance(results.get(key), str) and results[key].strip():
            return results[key].strip()
    return _run_prompt(results, best_id) or _read_text(EVOLVE_DIR / "prompts" / "best.txt")


# ---------------------------------------------------------------- before/after sample

def _sample(results, v0_prompt, best_prompt):
    """{author, stars, text, v0_reply, best_reply} for one held-out review, or None. Never raises."""
    if not v0_prompt or not best_prompt:
        return None
    key = hashlib.sha256(best_prompt.encode("utf-8")).hexdigest()[:16]
    try:
        saved = _read_json(SAMPLE_PATH)
        saved = saved if isinstance(saved, dict) else {}
        if isinstance(saved.get(key), dict):
            return saved[key]
        sample = _build_sample(results, v0_prompt, best_prompt)
        if sample:
            saved[key] = sample
            _write_json(SAMPLE_PATH, saved)
        return sample
    except Exception as e:  # store or LLM down (llm.LLMError), odd data: the card just has no sample
        print(f"warning: evolution sample unavailable: {type(e).__name__}")
        return None


def _build_sample(results, v0_prompt, best_prompt):
    split = _read_json(SPLIT_PATH) or {}
    reviews = _holdout_reviews(split.get("holdout_businesses") or [])
    holdout_ids = [rid for rid in split.get("holdout") or []
                   if (reviews.get(rid, {}).get("text") or "").strip()]
    if not holdout_ids:
        return None

    # 1. A review evolve already wrote both replies for: its generation cache, or results.json.
    #    Prefer the review where the best prompt gained the most points over v0.
    generations = _generations()
    model = config.get("LUMINAL_MODEL") or ""
    found = []
    for rid in holdout_ids:
        v0_reply = (generations.get(_generation_key(model, v0_prompt, rid))
                    or _holdout_detail(results, "v0", rid, "reply"))
        best_reply = (generations.get(_generation_key(model, best_prompt, rid))
                      or _holdout_detail(results, "best", rid, "reply"))
        if v0_reply and best_reply:
            gain = ((_holdout_detail(results, "best", rid, "score") or 0)
                    - (_holdout_detail(results, "v0", rid, "score") or 0))
            found.append((gain, rid, v0_reply, best_reply))
    if found:
        _, rid, v0_reply, best_reply = max(found, key=lambda f: f[0])
        return _sample_dict(reviews[rid], v0_reply, best_reply)

    # 2. Nothing cached: write both replies once for the first held-out review (2 LLM calls).
    review = reviews[holdout_ids[0]]
    return _sample_dict(review, _generate(v0_prompt, review, "v0"), _generate(best_prompt, review, "best"))


def _holdout_reviews(slugs):
    """{review_id: review} for the held-out businesses, each review with its business attached."""
    reviews = {}
    for slug in slugs:
        business = store.get_business(slug) or {}
        for review in store.reviews_for(slug):
            reviews[str(review["_id"])] = dict(review, business=business)
    return reviews


def _generations():
    """evolve's generation cache as {key: reply}. Bad lines (an append in progress) are skipped."""
    out = {}
    try:
        lines = GENERATIONS_PATH.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return out
    for line in lines:
        try:
            row = json.loads(line)
            out[row["k"]] = row["v"]
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _generation_key(model, prompt, review_id):
    """Same key as evolve/score.py generate(): sha256 of model, prompt, review id, temperature."""
    parts = (model, prompt, review_id, GEN_TEMPERATURE)
    return hashlib.sha256("\x00".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _holdout_detail(results, which, review_id, field):
    """results.json holdout_detail[which][review_id][field] for which in ("v0", "best"), or None."""
    detail = results.get("holdout_detail")
    rows = detail.get(which) if isinstance(detail, dict) else None
    row = rows.get(review_id) if isinstance(rows, dict) else None
    return row.get(field) if isinstance(row, dict) else None


def _review_message(review):
    """The user message evolve sends after a candidate prompt (mirrors evolve/score.py review_message)."""
    biz = review.get("business") or {}
    category = f" ({biz['category']})" if biz.get("category") else ""
    return (f"Business: {biz.get('name', review.get('business_id'))}{category}\n"
            f"Business phone: {biz.get('phone') or 'none listed'}\n"
            f"Reviewer: {review.get('author') or 'anonymous'}\n"
            f"Rating: {review['stars']}/5\n"
            f"Review:\n{review['text'].strip()}")


def _generate(prompt, review, which):
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": _review_message(review)}]
    return llm.complete(messages, tag=f"evolution-sample:{which}")["text"]


def _sample_dict(review, v0_reply, best_reply):
    return {
        "author": (review.get("author") or "").strip() or "Google user",
        "stars": int(review.get("stars") or 0),
        "text": (review.get("text") or "").strip(),
        "v0_reply": str(v0_reply).strip(),
        "best_reply": str(best_reply).strip(),
    }


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, locked, or half-written (JSONDecodeError is a ValueError)
        return None


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- chart

def chart_svg(rounds):
    """Inline SVG line chart of the best train score per round. rounds: [(round_number, score)]."""
    open_tag = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 120" width="100%" '
                'role="img" aria-label="{}" font-size="11">')
    if not rounds:
        return (open_tag.format("No rounds yet")
                + f'<text x="160" y="64" text-anchor="middle" fill="{MUTED}">No rounds yet</text></svg>')

    left, right, top, bottom = 40.0, 296.0, 14.0, 88.0
    scores = [score for _, score in rounds]
    lo, hi = min(scores), max(scores)

    def x(i):
        if len(rounds) == 1:
            return (left + right) / 2
        return left + i * (right - left) / (len(rounds) - 1)

    def y(score):
        if hi == lo:
            return (top + bottom) / 2
        return bottom - (score - lo) * (bottom - top) / (hi - lo)

    (first_round, first_score), (last_round, last_score) = rounds[0], rounds[-1]
    label = (f"Best training score per round: {first_score:.1f} in round {first_round}, "
             f"{last_score:.1f} in round {last_round}")
    parts = [open_tag.format(html.escape(label))]

    for score in sorted({lo, hi}):  # min and max: a dashed guide and a label
        parts.append(f'<line x1="{left}" y1="{y(score):.1f}" x2="{right}" y2="{y(score):.1f}" '
                     f'stroke="{GRID}" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{left - 8}" y="{y(score) + 4:.1f}" text-anchor="end" '
                     f'fill="{MUTED}">{score:.1f}</text>')

    points = [(x(i), y(score)) for i, (_, score) in enumerate(rounds)]
    if len(points) > 1:
        coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{INK}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
    for (px, py), (number, score) in zip(points, rounds):
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{DOT}" stroke="{INK}" '
                     f'stroke-width="1.5"><title>Round {number}: {score:.1f}</title></circle>')

    edge_labels = [(points[0][0], first_round)]
    if len(points) > 1:
        edge_labels.append((points[-1][0], last_round))
    for px, number in edge_labels:
        parts.append(f'<text x="{px:.1f}" y="110" text-anchor="middle" fill="{MUTED}">'
                     f'Round {number}</text>')
    parts.append("</svg>")
    return "".join(parts)
