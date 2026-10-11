"""Write replies for unanswered reviews: 40 threads, each reply saved the moment it returns.

Reviews with text go to the LLM; rating-only reviews get a short template (no invented details).
"""
import hashlib
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import llm
import prompts
import store

WORKERS = 40
PROGRESS_EVERY = 50
MIN_WORDS, MAX_WORDS = 25, 60

BANNED = re.compile(r"\b(valued|feedback|strive|apologies for any inconvenience|"
                    r"sorry you feel that way|polic(?:y|ies))\b", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
# offers of free things; "that's on us" (taking responsibility) is fine
OFFER = re.compile(r"\b(on the house|for free|free (?:meal|drink|round|dessert|appetizer|app)|discount|"
                   r"comped|complimentary|(?:it|one|another|next|round|meal|drinks?)\b[^.]{0,25}\bon us)\b", re.I)
PHONE_LINE = re.compile(r"call us at \[PHONE\]", re.I)

TEMPLATES = {
    "high": [
        "Thank you{name} for the great rating. It means a lot to me and the whole team, and I hope to see you back soon.",
        "Thanks{name} for taking a moment to rate us. I'm glad you enjoyed your visit, and I hope we see you again soon.",
        "Thank you{name} for the kind rating. It keeps a small place like ours going, and I look forward to your next visit.",
    ],
    "mid": [
        "Thank you{name} for coming in and for the rating. If anything could have been better, I'd like to hear it. "
        "Call us at [PHONE] and ask for the owner.",
        "Thanks{name} for stopping by. It sounds like we fell a little short, and I'd like to know where. "
        "Call us at [PHONE] and ask for the owner anytime.",
    ],
    "low": [
        "{lead}I'm sorry your visit fell short. I'd like to make it right, so please call us at [PHONE] and ask for the owner.",
        "{lead}this isn't the experience I want anyone to have with us, and I'm sorry. Please call us at [PHONE] "
        "and ask for the owner so I can make it right.",
    ],
}


def _digits(text):
    return re.sub(r"\D", "", text or "")


def first_name(author):
    """'Kevin T.' -> 'Kevin'; '' for initials and non-names like 'A Google User'."""
    words = (author or "").split()
    word = words[0].strip(".,") if words else ""
    if len(word) < 2 or not word.isalpha() or word.lower() in {"google", "user", "anonymous"}:
        return ""
    return word.capitalize() if word.isupper() or word.islower() else word


def template_reply(review):
    """Short, warm reply for a rating-only review, picked stably by review id. Keeps [PHONE]."""
    stars = review.get("stars") or 0
    bucket = "high" if stars >= 4 else "low" if 1 <= stars <= 2 else "mid"
    variants = TEMPLATES[bucket]
    text = variants[int(hashlib.sha1(review["_id"].encode("utf-8")).hexdigest(), 16) % len(variants)]
    name = first_name(review.get("author"))
    text = text.format(name=f", {name}," if name else "", lead=f"{name}, " if name else "")
    return text[0].upper() + text[1:]


def fill_phone(reply, phone):
    """Swap [PHONE] for the real number; with no number, ask them to stop by instead."""
    if phone:
        return reply.replace("[PHONE]", phone)
    reply = PHONE_LINE.sub(lambda m: "Stop by" if m.group(0)[0] == "C" else "stop by", reply)
    return reply.replace("[PHONE]", "us")


def problems(reply, review, phone):
    """Rule breaks that can be checked mechanically. Empty list = fine."""
    found = []
    words = len(reply.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        found.append(f"it has {words} words, it must have {MIN_WORDS} to {MAX_WORDS}")
    if reply.count("!") > 1:
        found.append("more than one exclamation mark")
    if EMOJI.search(reply) or re.search(r"#\w", reply):
        found.append("no emojis or hashtags")
    banned = sorted({b.lower() for b in BANNED.findall(reply)})
    if banned:
        found.append("do not use: " + ", ".join(banned))
    if OFFER.search(reply):
        found.append("do not offer anything free or discounted")
    if (review["stars"] <= 2 and phone and "[PHONE]" not in reply
            and _digits(phone) not in _digits(reply)):
        found.append('include "call us at [PHONE] and ask for the owner"')
    leftover = sorted(set(re.findall(r"\[[A-Z_ ]+\]", reply)) - {"[PHONE]"})
    if leftover:
        found.append("no placeholders like " + ", ".join(leftover))
    return found


def reply_for(review, business):
    """Write one reply: template for rating-only reviews, else the LLM plus one rule-fixing rewrite."""
    phone = business.get("phone") or ""
    if not review.get("text"):
        return {"text": fill_phone(template_reply(review), phone), "model": "template",
                "tokens_in": 0, "tokens_out": 0, "reasoning_tokens": 0, "problems": []}
    messages = prompts.build_messages(business, review)
    tag = review["_id"][:16]
    best = llm.complete(messages, tag=tag)
    totals = {k: best[k] for k in ("tokens_in", "tokens_out", "reasoning_tokens")}
    broken = problems(best["text"], review, phone)
    if broken:
        fix = messages + [{"role": "assistant", "content": best["text"]},
                          {"role": "user", "content": "Rewrite it. Fix: " + "; ".join(broken)
                           + ". Output only the reply text."}]
        try:
            second = llm.complete(fix, tag=tag + ":fix")
        except llm.LLMError:
            second = None
        if second:
            for k in totals:
                totals[k] += second[k]
            second_broken = problems(second["text"], review, phone)
            if len(second_broken) < len(broken):
                best, broken = second, second_broken
    return {"text": fill_phone(best["text"], phone), "model": best["model"], "problems": broken, **totals}


def _targets(slug, redo):
    if not redo:
        return store.pending_reviews(slug)
    slugs = [slug] if slug else [b["_id"] for b in store.list_businesses()]
    return [r for s in slugs for r in store.reviews_for(s) if not r["owner_replied"]]


def _progress(i, total, s, started):
    cost = (s["in"] + s["out"]) / 1_000_000 * config.LLM_COST_PER_MILLION_TOKENS
    print(f"[{i}/{total}] done {s['done']} | failed {s['failed']} | tokens {s['in']:,} in / {s['out']:,} out "
          f"(reasoning {s['reasoning']}) | est ${cost:.4f} | {time.time() - started:.0f}s", flush=True)


def generate(slug=None, redo=False, workers=WORKERS):
    """Write and save replies for one business, or all when slug is None. Returns the counters."""
    if slug and not store.get_business(slug):
        raise RuntimeError(f"unknown business {slug!r}; see: python run.py list")
    targets = _targets(slug, redo)
    if not targets:
        print("Nothing to write: every unanswered review has a reply (use --redo to rewrite them).")
        return {}
    businesses = {b["_id"]: b for b in store.list_businesses()}
    with_text = sum(1 for r in targets if r.get("text"))
    print(f"Writing {len(targets)} replies: {with_text} with {llm.model_name()}, "
          f"{len(targets) - with_text} rating-only templates, {workers} threads ...", flush=True)
    s = {"done": 0, "failed": 0, "flagged": 0, "in": 0, "out": 0, "reasoning": 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(reply_for, r, businesses[r["business_id"]]): r for r in targets}
        for i, future in enumerate(as_completed(futures), 1):
            review = futures[future]
            try:
                out = future.result()
                store.save_reply(review["_id"], out["text"], out["model"], out["tokens_in"], out["tokens_out"])
                s["done"] += 1
                s["in"] += out["tokens_in"]
                s["out"] += out["tokens_out"]
                s["reasoning"] += out["reasoning_tokens"]
                if out["problems"]:
                    s["flagged"] += 1
                    llm.log.warning("review=%s still breaks: %s", review["_id"], "; ".join(out["problems"]))
            except Exception as e:
                s["failed"] += 1
                llm.log.error("review=%s failed: %s: %s", review["_id"], type(e).__name__, e)
            if i % PROGRESS_EVERY == 0 or i == len(targets):
                _progress(i, len(targets), s, started)
    if s["failed"] or s["flagged"]:
        print(f"{s['failed']} failed, {s['flagged']} saved but still break a rule: details in data/llm.log")
    return s


def print_samples(slug, n=5):
    """Show n replies next to their reviews: a 1-star first if there is one, then 2, 3, 4-5 stars."""
    biz = store.get_business(slug)
    if not biz:
        raise RuntimeError(f"unknown business {slug!r}; see: python run.py list")
    phone = biz.get("phone") or ""
    answered = [r for r in store.reviews_for(slug) if r.get("reply") and not r["owner_replied"]]
    written = [r for r in answered if r.get("text")]
    picks = []
    for want in (1, 2, 3, 5, 4, 5, 4, 3, 2, 1):
        if len(picks) == n:
            break
        r = next((r for r in written if r["stars"] == want and r not in picks), None)
        if r:
            picks.append(r)
    picks += [r for r in answered if r not in picks][:n - len(picks)]

    print(f"\n=== {biz['name']} | {biz.get('category') or ''} | {phone or 'no phone'}")
    for r in picks:
        review = r["text"] if len(r["text"]) <= 420 else r["text"][:420].rsplit(" ", 1)[0] + " ..."
        print("-" * 96)
        print(f"{r['stars']}/5 stars | {r.get('author') or 'anonymous'} | {(r.get('date') or '')[:10]}"
              f" | {len(r['reply'].split())} words{' | template' if r.get('model') == 'template' else ''}")
        print(textwrap.fill(review or "(rating only, no text)", 96, initial_indent="REVIEW  ",
                            subsequent_indent="        "))
        print(textwrap.fill(r["reply"], 96, initial_indent="REPLY   ", subsequent_indent="        "))
    print("-" * 96)
    broken = [(r, problems(r["reply"], r, phone)) for r in written]
    broken = [(r, p) for r, p in broken if p]
    print(f"rule check: {len(written) - len(broken)}/{len(written)} LLM replies pass, "
          f"{len(answered) - len(written)} rating-only templates")
    for r, p in broken[:5]:
        print(f"  {r['stars']}/5 {r.get('author') or ''}: {'; '.join(p)}")
