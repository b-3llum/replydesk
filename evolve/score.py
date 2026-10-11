"""Scores a reply prompt: generate a reply per review, then rate each (review, reply) pair 0-100.

    python -m evolve.score                                  # v0 on the training split
    python -m evolve.score --prompt-file evolve/prompts/best.txt
    python -m evolve.score --set holdout --allow-holdout    # final evaluation only

Points per pair:
    15  word count 25-60
    15  shares a content word with the review (see CONTENT WORDS below)
    10  uses the reviewer's first name (free when the reviewer name isn't a first name)
    15  1-2 stars: business phone + a direct offer to fix it (offer only if no phone on file)
        4-5 stars: an invitation back.  3 stars: either one
    10  no banned phrases, no emojis, at most one "!"
    35  LLM judge 0-10 x 3.5: strict-editor rubric, temperature 0, cached by hash of
        (JUDGE_VERSION, review_id, reply)

Generations are cached too (model + prompt + review + temperature), so rescoring is free.
"""
import argparse
import hashlib
import json
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evolve import llm_client
from evolve.common import CACHE_DIR, PROMPTS_DIR, SPLIT_PATH, db, has_text

GEN_TEMPERATURE = 0.3
GEN_MAX_TOKENS = 300
WORKERS = 8

BANNED = ("valued", "feedback", "strive", "apologies for any inconvenience", "sorry you feel")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️‍]")
OFFER = re.compile(
    r"make (it|this|things) right|fix (it|this|that)|set (it|this|things) right|call (me|us)"
    r"|give (me|us) a (call|ring)|reach (out|me|us)|contact (me|us)|ask for me"
    r"|(on|on the) (us|me|the house)|refund|replace|redo|remake|another chance|(talk|speak) (with|to) you"
    r"|(like|love|want) to (hear|talk|speak|chat)|come back in and|next (meal|order|visit) is",
    re.I)
INVITE = re.compile(
    r"come back|see you|next time|come again|visit again|stop (by|in)|back soon|swing by"
    r"|hope to (see|have|welcome)|welcom(e|ing) you|you back|return|again soon|drop (by|in)"
    r"|come (in|by) again|next visit|look forward to (seeing|having|serving)",
    re.I)

# CONTENT WORDS: there's no POS tagger in this venv (and the shared requirements are off
# limits), so "noun" is approximated: lowercase alphabetic tokens of length >= 4 that are not
# stopwords, not generic review/reply vocabulary (thank, food, place, great, service...),
# and not words from the business's own name or category, which would match trivially.
STOPWORDS = set("""
about above after again against also always among another anyone anything around away back
because been before being below between both came come comes coming could didn doesn doing done
down during each either else even ever every from further gets getting give given goes going gone
have having here hers herself himself into itself just keep know last less like little made make
many maybe more most much must myself never next none once only other ours ourselves over own
really same shall should since some something still such sure take than that their theirs them
themselves then there these they thing things this those though through thus till together
too took toward under until upon very want wants were what whatever when where whether which
while whom whose will with within without would your yours yourself yourselves went well
okay also again ever been dont cant wont isnt wasnt arent werent youre theyre thats
""".split())
GENERIC = set("""
thank thanks thankful review reviews reviewer place spot food restaurant experience experiences
visit visits visited visiting time times great good best better amazing awesome delicious
excellent wonderful fantastic nice love loved lovely enjoy enjoyed enjoying recommend recommended
highly definitely absolutely truly totally service staff team owner owners customer customers
guest guests star stars everything everyone anyone someone people today tonight sorry glad happy
hear heard hope hoping appreciate appreciated kind words taking share sharing hearing welcome
order ordered back soon again more much day days week weeks year years first last whole
""".split())


def _norm(word):
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def content_words(text, exclude=()):
    return {_norm(w) for w in re.findall(r"[a-z]+", (text or "").lower().replace("’", "'"))
            if len(w) >= 4 and w not in STOPWORDS and w not in GENERIC} - set(exclude)


def first_name(author):
    """'Firstname L' -> 'Firstname'; handles like 'Taco-lover' or 'foodie123' -> None."""
    parts = (author or "").split()
    if parts and re.fullmatch(r"[A-Z][a-z]+(?:'[A-Za-z]+)?", parts[0]):
        return parts[0]
    return None


def phone_digits(phone):
    digits = re.sub(r"\D", "", phone or "")
    return digits[-10:] if len(digits) >= 10 else None


# ---------------------------------------------------------------- data

_reviews = {}
_reviews_lock = threading.Lock()


def load_reviews(ids):
    """{id: review} with the business attached as review['business']."""
    missing = [i for i in ids if i not in _reviews]
    if missing:
        rows = list(db().reviews.find({"_id": {"$in": missing}}))
        bizs = {b["_id"]: b for b in db().businesses.find(
            {"_id": {"$in": list({r["business_id"] for r in rows})}})}
        with _reviews_lock:
            for r in rows:
                if not has_text(r):
                    raise ValueError(f"review {r['_id']} has no text")
                r["business"] = bizs.get(r["business_id"], {})
                _reviews[r["_id"]] = r
    lost = [i for i in ids if i not in _reviews]
    if lost:
        raise KeyError(f"reviews not in Mongo: {lost[:3]}...")
    return {i: _reviews[i] for i in ids}


def load_split():
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- caches (append-only jsonl)

class JsonlCache:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.data[row["k"]] = row["v"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


GEN_CACHE = JsonlCache(CACHE_DIR / "generations.jsonl")
JUDGE_CACHE = JsonlCache(CACHE_DIR / "judge.jsonl")


def _hash(*parts):
    return hashlib.sha256("\x00".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- generation

def review_message(review):
    biz = review["business"]
    category = f" ({biz['category']})" if biz.get("category") else ""
    return (f"Business: {biz.get('name', review['business_id'])}{category}\n"
            f"Business phone: {biz.get('phone') or 'none listed'}\n"
            f"Reviewer: {review.get('author') or 'anonymous'}\n"
            f"Rating: {review['stars']}/5\n"
            f"Review:\n{review['text'].strip()}")


def _clean(reply):
    reply = reply.strip()
    if len(reply) >= 2 and reply[0] == reply[-1] and reply[0] in "\"'":
        reply = reply[1:-1].strip()
    return reply


def generate(prompt_text, review, temperature=GEN_TEMPERATURE):
    key = _hash(llm_client.MODEL, prompt_text, review["_id"], temperature)
    cached = GEN_CACHE.get(key)
    if cached is not None:
        return cached
    reply, _ = llm_client.chat(
        [{"role": "system", "content": prompt_text},
         {"role": "user", "content": review_message(review)}],
        temperature=temperature, max_tokens=GEN_MAX_TOKENS, tag="generate")
    reply = _clean(reply)
    GEN_CACHE.put(key, reply)
    return reply


# ---------------------------------------------------------------- judge

# Bump whenever JUDGE_PROMPT or the judge's parsing changes: it is part of the cache key,
# so scores from an older judge are never reused.
JUDGE_VERSION = "v2-strict-editor"
JUDGE_PROMPT = """You are a strict editor for a local business owner. Score this reply to a Google review from 0 to 10. Start at 10 and subtract:
-3 if it has any preamble, heading, markdown, quotes around the reply, or anything that is not the reply itself
-2 if it sounds corporate or templated (words like feedback, valued, strive, team, experience, apologies for any inconvenience, we take this seriously)
-2 if it does not mention one concrete detail from this specific review
-2 if it is longer than 60 words or shorter than 25
-1 if the tone is wrong for the star rating (gushing on a 1-star, defensive on a complaint, flat on a 5-star)
Minimum 0. Reply with only the number.

Business: {name}{category}
Rating: {stars}/5
Review by {author}:
{text}

Reply:
{reply}"""


def judge(review, reply):
    """0-10 integer, cached by hash of (JUDGE_VERSION, review_id, reply)."""
    if not reply:
        return 0.0
    key = _hash(JUDGE_VERSION, review["_id"], reply)
    cached = JUDGE_CACHE.get(key)
    if cached is not None:
        return cached
    biz = review["business"]
    text, _ = llm_client.chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(
            name=biz.get("name", ""), category=f" ({biz['category']})" if biz.get("category") else "",
            stars=review["stars"], author=review.get("author") or "anonymous",
            text=review["text"].strip(), reply=reply)}],
        temperature=0, max_tokens=8, tag="judge")
    m = re.search(r"-?\d+", text)
    if not m:
        raise ValueError(f"judge returned no number: {text!r}")
    value = float(max(0, min(10, int(m.group()))))
    JUDGE_CACHE.put(key, value)
    return value


# ---------------------------------------------------------------- scoring

def deterministic_checks(review, reply):
    """{check: passed} for the 65 deterministic points."""
    biz = review["business"]
    words = re.findall(r"[A-Za-z0-9’']+", reply)
    exclude = content_words(f"{biz.get('name', '')} {biz.get('category', '')}")
    name = first_name(review.get("author"))
    stars = int(review["stars"])
    digits = phone_digits(biz.get("phone"))
    has_phone = digits is None or digits in re.sub(r"\D", "", reply)
    offer = bool(OFFER.search(reply))
    invite = bool(INVITE.search(reply))
    lowered = reply.lower()
    if stars <= 2:
        outcome = has_phone and offer
    elif stars >= 4:
        outcome = invite
    else:
        outcome = invite or offer
    return {
        "length": 25 <= len(words) <= 60,
        "specific": bool(content_words(reply, exclude) & content_words(review["text"], exclude)),
        "name": name is None or re.search(rf"\b{re.escape(name)}\b", reply, re.I) is not None,
        "outcome": outcome,
        "clean": (not any(b in lowered for b in BANNED) and not EMOJI.search(reply)
                  and reply.count("!") <= 1),
    }


POINTS = {"length": 15, "specific": 15, "name": 10, "outcome": 15, "clean": 10}


def score_pair(review, reply):
    checks = deterministic_checks(review, reply)
    j = judge(review, reply)
    score = sum(POINTS[k] for k, ok in checks.items() if ok) + j * 3.5
    return {"score": round(score, 2), "judge": j, "checks": checks,
            "failed": [k for k, ok in checks.items() if not ok]}


def score_prompt(prompt_text, review_ids, workers=WORKERS):
    """Mean 0-100 score of a prompt over review_ids, per-review detail, and the 5 worst pairs."""
    reviews = load_reviews(review_ids)

    def one(rid):
        review = reviews[rid]
        reply = generate(prompt_text, review)
        return rid, dict(score_pair(review, reply), reply=reply)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_review = dict(pool.map(one, review_ids))
    worst = sorted(review_ids, key=lambda rid: per_review[rid]["score"])[:5]
    return {
        "mean": round(statistics.mean(r["score"] for r in per_review.values()), 2),
        "per_review": per_review,
        "worst": [{"review_id": rid, "stars": reviews[rid]["stars"],
                   "author": reviews[rid].get("author"), "review": reviews[rid]["text"].strip(),
                   "reply": per_review[rid]["reply"], "score": per_review[rid]["score"],
                   "judge": per_review[rid]["judge"], "failed": per_review[rid]["failed"]}
                  for rid in worst],
    }


def summarize(result):
    rows = result["per_review"].values()
    n = len(rows)
    rates = {k: sum(1 for r in rows if r["checks"][k]) / n for k in POINTS}
    judge_mean = statistics.mean(r["judge"] for r in rows)
    return rates, judge_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default=str(PROMPTS_DIR / "v0.txt"))
    ap.add_argument("--set", choices=["train", "holdout"], default="train")
    ap.add_argument("--allow-holdout", action="store_true", help="required for --set holdout")
    ap.add_argument("--show", type=int, default=5, help="worst pairs to print")
    args = ap.parse_args()
    if args.set == "holdout" and not args.allow_holdout:
        raise SystemExit("held-out reviews are for the final evaluation; pass --allow-holdout")
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    ids = load_split()[args.set]
    result = score_prompt(prompt_text, ids)
    rates, judge_mean = summarize(result)
    print(f"{Path(args.prompt_file).name} on {args.set} ({len(ids)} reviews): mean {result['mean']}")
    print("  pass rates: " + "  ".join(f"{k} {v:.0%}" for k, v in rates.items())
          + f"  | judge mean {judge_mean:.2f}/10")
    for w in result["worst"][:args.show]:
        print(f"\n  [{w['score']:5.1f}] {w['stars']}* {w['author']} | failed: {', '.join(w['failed']) or '-'}"
              f" | judge {w['judge']}")
        print("    review: " + w["review"][:220].replace("\n", " "))
        print("    reply:  " + w["reply"][:400].replace("\n", " "))
    t = llm_client.TOTALS
    print(f"\nLLM calls this run: {t['calls']} (cache misses), prompt tokens {t['prompt_tokens']}, "
          f"completion tokens {t['completion_tokens']}, reasoning tokens {t['reasoning_tokens']}")


if __name__ == "__main__":
    main()
