"""Write replies for unanswered reviews: 40 threads, each reply saved the moment it returns.

Reviews with text go to the LLM; rating-only reviews get a short template (no invented details).
Then each business gets a diversity pass: no two replies share their first four words or their
last sentence, and the 1-2 star replies phrase the call line at least three ways.
"""
import hashlib
import re
import textwrap
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import llm
import prompts
import store

WORKERS = 40
PROGRESS_EVERY = 50
MIN_WORDS, MAX_WORDS = 25, 60
CALL_LINE_PHRASINGS = 3

BANNED = re.compile(r"\b(valued|feedback|strive|apologies for any inconvenience|"
                    r"sorry you feel that way|polic(?:y|ies))\b", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
# offers of free things; "that's on us" (taking responsibility) is fine
OFFER = re.compile(r"\b(on the house|for free|free (?:meal|drink|round|dessert|appetizer|app)|discount|"
                   r"comped|complimentary|(?:it|one|another|next|round|meal|drinks?)\b[^.]{0,25}\bon us)\b", re.I)
NO_PHONE = re.compile(r"\b(?:call us at|call me at|reach us at|reach me at|call) \[PHONE\]", re.I)
PHONE_NUMBER = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")

TEMPLATES = {
    "high": [
        "Thank you{name} for the great rating. It means a lot, and I hope to see you back soon.",
        "Thanks{name} for taking a moment to rate us. I'm glad you enjoyed your visit, and I hope we see you again soon.",
        "Thank you{name} for the kind rating. Ratings like this keep a small place like ours going. "
        "I look forward to your next visit.",
        "{lead}I appreciate you taking the time to rate us. It made my day. Come back and see us soon.",
    ],
    "mid": [
        "Thank you{name} for coming in and for the rating. If anything could have been better, I'd like to hear it. "
        "Call us at [PHONE] and ask for the owner.",
        "Thanks{name} for stopping by. It sounds like we fell a little short, and I'd like to know where. "
        "Call [PHONE] and ask for the owner anytime.",
    ],
    "low": [
        "{lead}I'm sorry your visit fell short. I'd like to make it right, so please call us at [PHONE] and ask for the owner.",
        "{lead}this isn't the experience I want anyone to have with us, and I'm sorry. You can reach me at [PHONE], "
        "just ask for the owner, and I'll make it right.",
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


def _bucket(review):
    stars = review.get("stars") or 0
    return "high" if stars >= 4 else "low" if 1 <= stars <= 2 else "mid"


def template_reply(review, variant=None):
    """Short, warm reply for a rating-only review; keeps [PHONE] for fill_phone.
    variant rotates the wording within a business; without it the review id picks one."""
    variants = TEMPLATES[_bucket(review)]
    if variant is None:
        variant = int(hashlib.sha1(review["_id"].encode("utf-8")).hexdigest(), 16)
    text = variants[variant % len(variants)]
    name = first_name(review.get("author"))
    text = text.format(name=f", {name}," if name else "", lead=f"{name}, " if name else "")
    return text[0].upper() + text[1:]


def fill_phone(reply, phone):
    """Swap [PHONE] for the real number; with no number, ask them to stop by instead."""
    if phone:
        return reply.replace("[PHONE]", phone)
    reply = NO_PHONE.sub(lambda m: "Stop by" if m.group(0)[0].isupper() else "stop by", reply)
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


def reply_for(review, business, variant=None):
    """Write one reply: template for rating-only reviews, else the LLM plus one rule-fixing rewrite."""
    phone = business.get("phone") or ""
    if not review.get("text"):
        return {"text": fill_phone(template_reply(review, variant), phone), "model": "template",
                "tokens_in": 0, "tokens_out": 0, "reasoning_tokens": 0, "problems": []}
    messages = prompts.build_messages(business, review)
    tag = review["_id"][-12:]  # Google review ids share long prefixes; the end is unique
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


# ---------- diversity within a business ----------

def _norm(text):
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def opening_key(reply):
    """First four words, lowercased, punctuation dropped."""
    return " ".join(_norm(reply).split()[:4])


def closing_key(reply):
    """Last sentence, lowercased, punctuation dropped."""
    parts = _sentences(reply)
    return _norm(parts[-1]) if parts else ""


def call_key(reply):
    """How the call line is worded: up to 3 words before the phone number and 5 after, in its sentence."""
    marked = PHONE_NUMBER.sub(" zzphone ", (reply or "").replace("[PHONE]", " zzphone "))
    for sentence in _sentences(marked):
        words = _norm(sentence).split()
        if "zzphone" in words:
            i = words.index("zzphone")
            return " ".join(words[max(0, i - 3):i + 6])
    return None


def _taken(text, rid, key, among=None):
    """key() of every other reply, or of the other replies in among."""
    return {key(text[o]) for o in (text if among is None else among) if o != rid} - {None, ""}


def _quoted(values):
    return "; ".join(f'"{v}"' for v in sorted(values))


def _rewrite(biz, review, text, asks, low_ids):
    """One targeted rewrite for the asked fixes ("opening", "closing", "call"). Keeps whichever
    version has fewer issues, saves it with the extra tokens, returns True if the rewrite won."""
    rid, phone = review["_id"], biz.get("phone") or ""

    def issues(reply):
        n = len(problems(reply, review, phone))
        if "opening" in asks and opening_key(reply) in _taken(text, rid, opening_key):
            n += 1
        if "closing" in asks and closing_key(reply) in _taken(text, rid, closing_key):
            n += 1
        if "call" in asks and (call_key(reply) is None or call_key(reply) in _taken(text, rid, call_key, low_ids)):
            n += 1
        return n

    fixes = []
    if "opening" in asks:
        fixes.append("Start it differently: its first four words must not match any of these openings that "
                     "other replies already use: " + _quoted(_taken(text, rid, opening_key)) + ".")
    if "closing" in asks:
        fixes.append("End it with a different last sentence than any of these: "
                     + _quoted(_taken(text, rid, closing_key)) + ".")
    if "call" in asks:
        fixes.append("Word the line asking them to call differently from these, which other replies already "
                     "use: " + _quoted(_taken(text, rid, call_key, low_ids)) + f". Keep the number "
                     f"{phone or '[PHONE]'} and keep it a direct way to reach the owner.")
    messages = prompts.build_messages(biz, review) + [
        {"role": "assistant", "content": text[rid]},
        {"role": "user", "content": "Rewrite it with the same facts and tone. " + " ".join(fixes)
                                    + " Output only the reply text."}]
    try:
        out = llm.complete(messages, tag=rid[-12:] + ":vary")
    except llm.LLMError as e:
        llm.log.error("review=%s vary rewrite failed: %s", rid, e)
        return False
    new = fill_phone(out["text"], phone)
    won = issues(new) < issues(text[rid])
    if won:
        text[rid] = new
        review["generated_at"] = None  # new text: save_reply stamps it now
    review["tokens_in"] = (review.get("tokens_in") or 0) + out["tokens_in"]
    review["tokens_out"] = (review.get("tokens_out") or 0) + out["tokens_out"]
    # A rewrite that lost keeps the reply's date, so it doesn't count as written today in the admin.
    store.save_reply(rid, text[rid], review.get("model") or out["model"], review["tokens_in"], review["tokens_out"],
                     generated_at=review.get("generated_at"))
    return won


def diversify(slug):
    """One business: unique first four words and last sentence for every LLM reply, and at least
    3 call-line wordings across its 1-2 star replies. At most one rewrite per reply. Rating-only
    templates are fixed text: they count as taken but are never rewritten."""
    biz = store.get_business(slug)
    rows = [r for r in store.reviews_for(slug) if r.get("reply") and not r["owner_replied"]]
    text = {r["_id"]: r["reply"] for r in rows}
    written = [r for r in rows if r.get("model") != "template"]
    low_ids = [r["_id"] for r in rows if r["stars"] <= 2]
    need = min(CALL_LINE_PHRASINGS, len(low_ids)) if biz.get("phone") else 0  # no phone, no call line

    def call_wordings():
        return len({call_key(text[i]) for i in low_ids} - {None})

    fixed, touched = Counter(), set()
    seen_open = {opening_key(text[r["_id"]]) for r in rows if r.get("model") == "template"}
    seen_close = {closing_key(text[r["_id"]]) for r in rows if r.get("model") == "template"}
    for r in written:  # newest first: the earlier of two twins keeps its wording
        rid = r["_id"]
        asks = []
        if opening_key(text[rid]) in seen_open:
            asks.append("opening")
        if closing_key(text[rid]) in seen_close:
            asks.append("closing")
        if asks and r["stars"] <= 2 and call_wordings() < need:
            asks.append("call")
        if asks:
            touched.add(rid)
            if _rewrite(biz, r, text, asks, low_ids):
                fixed.update(asks)
        seen_open.add(opening_key(text[rid]))
        seen_close.add(closing_key(text[rid]))
    while call_wordings() < need:
        groups = Counter(call_key(text[i]) for i in low_ids)
        r = next((r for r in written if r["stars"] <= 2 and r["_id"] not in touched
                  and groups[call_key(text[r["_id"]])] > 1), None)
        if r is None:
            break
        touched.add(r["_id"])
        if _rewrite(biz, r, text, ["call"], low_ids):
            fixed["call"] += 1
    fixed["trimmed"] += _trim_twin_closings(biz, written, text)
    return fixed


def _trim_twin_closings(biz, written, text):
    """Last resort for twins the rewrite could not separate: drop the repeated last sentence when
    what is left still follows every rule and ends uniquely. Mechanical, no LLM call."""
    phone, trimmed = biz.get("phone") or "", 0
    for r in written:
        rid = r["_id"]
        parts = _sentences(text[rid])
        if len(parts) < 2 or closing_key(text[rid]) not in _taken(text, rid, closing_key):
            continue
        shorter = " ".join(parts[:-1])
        if problems(shorter, r, phone) or closing_key(shorter) in _taken(text, rid, closing_key):
            continue
        text[rid] = shorter
        store.save_reply(rid, shorter, r.get("model"), r.get("tokens_in") or 0, r.get("tokens_out") or 0,
                         generated_at=r.get("generated_at"))  # no LLM call: the reply keeps its date
        trimmed += 1
    return trimmed


def _diversify_safe(slug):
    try:
        return diversify(slug)
    except Exception as e:
        llm.log.error("diversify %s failed: %s: %s", slug, type(e).__name__, e)
        return Counter()


def diversity_report(slug):
    """What is still repeated for one business (LLM replies only; templates may repeat each other)."""
    has_phone = bool((store.get_business(slug) or {}).get("phone"))
    rows = [r for r in store.reviews_for(slug) if r.get("reply") and not r["owner_replied"]]
    text = {r["_id"]: r["reply"] for r in rows}
    written = [r["_id"] for r in rows if r.get("model") != "template"]
    low_ids = [r["_id"] for r in rows if r["stars"] <= 2]
    return {
        "openings_repeated": sum(1 for i in written if opening_key(text[i]) in _taken(text, i, opening_key)),
        "closings_repeated": sum(1 for i in written if closing_key(text[i]) in _taken(text, i, closing_key)),
        "call_wordings": len({call_key(text[i]) for i in low_ids} - {None}),
        "call_wordings_needed": min(CALL_LINE_PHRASINGS, len(low_ids)) if has_phone else 0,
    }


# ---------- batch ----------

def _targets(slug, redo):
    if not redo:
        targets = store.pending_reviews(slug)
    else:
        slugs = [slug] if slug else [b["_id"] for b in store.list_businesses()]
        targets = [r for s in slugs for r in store.reviews_for(s) if not r["owner_replied"]]
    return sorted(targets, key=lambda r: r.get("date") or "", reverse=True)


def _progress(i, total, s, started):
    cost = (s["in"] + s["out"]) / 1_000_000 * config.LLM_COST_PER_MILLION_TOKENS
    print(f"[{i}/{total}] done {s['done']} | failed {s['failed']} | tokens {s['in']:,} in / {s['out']:,} out "
          f"(reasoning {s['reasoning']}) | est ${cost:.4f} | {time.time() - started:.0f}s", flush=True)


def generate(slug=None, redo=False, workers=WORKERS):
    """Write and save replies for one business, or all when slug is None, then run the diversity pass."""
    if slug and not store.get_business(slug):
        raise RuntimeError(f"unknown business {slug!r}; see: python run.py list")
    targets = _targets(slug, redo)
    if not targets:
        print("Nothing to write: every unanswered review has a reply (use --redo to rewrite them).")
        _diversity_pass([slug] if slug else [b["_id"] for b in store.list_businesses()], time.time())
        return {}
    businesses = {b["_id"]: b for b in store.list_businesses()}
    orphans = sum(1 for r in targets if r.get("business_id") not in businesses)
    if orphans:  # e.g. an interrupted scrape: it saves reviews before their business
        print(f"Skipping {orphans} review(s) whose business is not in the store.", flush=True)
        targets = [r for r in targets if r.get("business_id") in businesses]
    variants, used = {}, Counter()
    for r in targets:  # rotate template wordings within each business
        if not r.get("text"):
            key = (r["business_id"], _bucket(r))
            variants[r["_id"]] = used[key]
            used[key] += 1
    with_text = sum(1 for r in targets if r.get("text"))
    print(f"Writing {len(targets)} replies: {with_text} with {llm.model_name()}, "
          f"{len(targets) - with_text} rating-only templates, {workers} threads ...", flush=True)
    s = {"done": 0, "failed": 0, "flagged": 0, "in": 0, "out": 0, "reasoning": 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(reply_for, r, businesses[r["business_id"]], variants.get(r["_id"])): r
                   for r in targets}
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
    _diversity_pass(sorted({r["business_id"] for r in targets}), started)
    return s


def _diversity_pass(slugs, started):
    if not slugs:  # nothing to vary, e.g. an empty store
        return
    print(f"Diversity pass over {len(slugs)} business(es) ...", flush=True)
    with ThreadPoolExecutor(max_workers=min(10, len(slugs))) as pool:
        fixed = sum(pool.map(_diversify_safe, slugs), Counter())
    left = [diversity_report(x) for x in slugs]
    print(f"diversity: rewrote {fixed['opening']} openings, {fixed['closing']} closings, {fixed['call']} call lines, "
          f"trimmed {fixed['trimmed']} repeated last sentences "
          f"| still repeated: {sum(d['openings_repeated'] for d in left)} openings, "
          f"{sum(d['closings_repeated'] for d in left)} closings | businesses short on call-line wordings: "
          f"{sum(1 for d in left if d['call_wordings'] < d['call_wordings_needed'])} "
          f"| {time.time() - started:.0f}s", flush=True)


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
    d = diversity_report(slug)
    print(f"diversity: {d['openings_repeated']} LLM replies repeat an opening, {d['closings_repeated']} repeat a "
          f"closing | call-line wordings across 1-2 star replies: {d['call_wordings']} "
          f"(need {d['call_wordings_needed']})")
