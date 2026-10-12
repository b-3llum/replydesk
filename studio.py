"""Reply Studio (/studio): list rows, review detail, LLM tags and the AI reply composer.

Nothing here posts to Google or changes a saved reply: compose() only returns text for the owner to
copy. The one write is store.save_tags, which caches a review's tags after their first extraction.
"""
import collections
import json
import re
import threading
from datetime import datetime

import generate
import llm
import prompts
import quality
import store

SOURCE = "Google Reviews"
SNIPPET_CHARS = 110
RATING_ONLY = "Rated only, no written review."
LABELS = {"negative": "Negative", "neutral": "Unanswered", "positive": "Positive"}  # the pill; studio.html's filters match
GROUPS = ("negative", "neutral", "positive")  # list order: complaints first
DEFAULT_TONES = {"negative": "empathetic", "neutral": "professional", "positive": "warm"}
ACTIONS = ("regenerate", "shorten")
MAX_TAGS, MAX_TAG_CHARS = 4, 32

# Appended to the review message on Regenerate; prompts.SYSTEM's rules still apply.
TONES = {
    "empathetic": "Tone: empathetic. Start by acknowledging how they felt, in plain words, before anything else.",
    "professional": "Tone: professional. Polished and courteous, no slang, still in first person as the owner.",
    "warm": "Tone: warm. Friendly and personal, like a neighbor who runs the place.",
    "concise": "Tone: concise. 25 to 35 words, one or two sentences.",
}
NO_TEXT = ("This is a star rating with no written review, so do not mention or invent any detail "
           "of their visit.")
SHORTEN = ("Shorten this reply to {least} to {most} words. Keep the reviewer's first name, the specific "
           "detail from their review and any phone line, still in first person as the owner. "
           "Output only the reply.\n\nReply: {reply}")
WORD = re.compile(r"[A-Za-z0-9’']+")  # the quality check's word rule
TAGS_SYSTEM = "You tag customer reviews for a small business owner. Output only JSON."
TAGS_PROMPT = """Tag this {stars}-star Google review of {name} ({category}).
Give 1 to 4 tags of 2 to 4 words each naming what the review is about, for example "Long wait time", "Food quality", "Friendly staff".
kind is "problem" when the tag is a complaint or something that went wrong, otherwise "topic".
Use only what the review says. Output only a JSON array like:
[{{"label": "Long wait time", "kind": "problem"}}, {{"label": "Friendly staff", "kind": "topic"}}]

Review: {text}"""


def kind(stars):
    """negative for 1-2 stars, positive for 4-5, neutral otherwise (3): the same buckets as generate."""
    if stars >= 4:
        return "positive"
    return "negative" if 1 <= stars <= 2 else "neutral"


def default_tone(stars):
    return DEFAULT_TONES[kind(stars)]


def _has_text(review):
    return bool((review.get("text") or "").strip())


def _date(iso):
    """'2026-10-03T15:22:51+00:00' -> 'Oct 3, 2026' (as on the /p cards); '' when missing."""
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return ""
    return f"{dt:%b} {dt.day}, {dt.year}"


def snippet(text):
    """The review on one line, cut at a word to about SNIPPET_CHARS characters with an ellipsis."""
    text = " ".join((text or "").split())
    if not text:
        return RATING_ONLY
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0].rstrip(" ,.;:!?-") + "…"


def row(review):
    """One list row: who, stars, kind and label for the pill, a one-line snippet and the date."""
    author = (review.get("author") or "").strip() or "Google user"
    stars = int(review.get("stars") or 0)
    k = kind(stars)
    return {"id": review["_id"], "author": author, "initial": author[0].upper(), "stars": stars,
            "kind": k, "label": LABELS[k], "snippet": snippet(review.get("text")),
            "date": _date(review.get("date"))}


def rows(reviews):
    """Rows for reviews given newest first: negative, then neutral, then positive, newest first in each."""
    return sorted((row(r) for r in reviews), key=lambda r: GROUPS.index(r["kind"]))  # stable sort


def counts(items):
    tally = collections.Counter(r["kind"] for r in items)
    return {"all": len(items), "negative": tally["negative"], "positive": tally["positive"],
            "neutral": tally["neutral"]}


def saved_tags(review):
    """Tags saved on the review; [] for a rating-only review (nothing to tag); None if not extracted yet."""
    if review.get("tags") is not None:
        return review["tags"]
    return None if _has_text(review) else []


def detail(review, business):
    """review.json: the whole review, its saved reply, tags, the default tone and the reply's quality."""
    out = row(review)
    del out["snippet"]
    reply = review.get("reply") or ""
    out.update(text=(review.get("text") or "").strip(), source=SOURCE, reply=reply, tags=saved_tags(review),
               tone=default_tone(out["stars"]), quality=quality.report(review, business, reply))
    return out


# ---------------------------------------------------------------- tags

_tag_locks = {}
_tag_locks_guard = threading.Lock()
_tags_done = {}  # review id -> tags this process extracted, for a request that waited on the lock


def tags_for(business, review):
    """tags.json: the saved tags, or ONE LLM call that extracts them and saves them on the review doc.
    [] when the call or its JSON fails; nothing is saved then, so the next open tries again."""
    tags = saved_tags(review)
    if tags is not None:
        return tags
    rid = review["_id"]
    with _tag_locks_guard:
        lock = _tag_locks.setdefault(rid, threading.Lock())
    with lock:  # a second open of the same review waits for the first call instead of paying again
        if rid in _tags_done:
            return _tags_done[rid]
        tags = extract_tags(business, review)
        if tags:
            store.save_tags(rid, tags)
            _tags_done[rid] = tags
        return tags


def extract_tags(business, review):
    """1 to 4 {"label", "kind"} tags from one cheap LLM call; [] if the call or its JSON fails."""
    text = (review.get("text") or "").strip()
    if not text:
        return []
    if len(text) > prompts.MAX_REVIEW_CHARS:
        text = text[:prompts.MAX_REVIEW_CHARS].rsplit(" ", 1)[0] + " ..."
    messages = [{"role": "system", "content": TAGS_SYSTEM},
                {"role": "user", "content": TAGS_PROMPT.format(
                    stars=review.get("stars"), name=business.get("name") or "a local business",
                    category=business.get("category") or "local business", text=text)}]
    try:
        out = llm.complete(messages, tag=f"studio:tags:{review['_id'][-12:]}")
    except llm.LLMError as e:
        llm.log.error("review=%s studio tags failed: %s", review["_id"], e)
        return []
    tags = parse_tags(out["text"])
    if not tags:
        llm.log.warning("review=%s studio tags not parsable: %r", review["_id"], out["text"][:200])
    return tags


def parse_tags(text):
    """The first JSON array in text that holds valid tags -> at most MAX_TAGS of them, else []."""
    decoder = json.JSONDecoder()
    start = (text or "").find("[")
    while start >= 0:
        try:
            items, _ = decoder.raw_decode(text, start)
        except ValueError:
            items = None
        tags = _valid_tags(items) if isinstance(items, list) else []
        if tags:
            return tags
        start = text.find("[", start + 1)
    return []


def _valid_tags(items):
    tags, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            continue
        label = " ".join(item["label"].split()).strip(" .,;:!\"'")
        if not label or len(label) > MAX_TAG_CHARS or label.lower() in seen:
            continue
        seen.add(label.lower())
        problem = str(item.get("kind") or "").strip().lower() == "problem"
        tags.append({"label": label[0].upper() + label[1:], "kind": "problem" if problem else "topic"})
    return tags[:MAX_TAGS]


# ---------------------------------------------------------------- composer

def _words(text):
    return len(WORD.findall(text or ""))


def shorten_target(current):
    """(least, most) words for Shorten: 25 to 35 for a long reply, about three quarters of a reply
    that is already short, so every click makes it shorter."""
    most = max(8, min(35, _words(current) * 3 // 4))
    return min(25, max(5, most - 10)), most


def compose(business, review, action, tone, text):
    """compose.json: a new reply, never saved. regenerate writes one in the tone, plus one rule-fixing
    rewrite like generate.reply_for; shorten trims the given text, with one more try if the answer is
    not shorter. Raises llm.LLMError."""
    phone = business.get("phone") or ""
    tag = f"studio:{action}:{review['_id'][-12:]}"
    if action == "shorten":
        current = (text or "").strip() or review.get("reply") or ""
        least, most = shorten_target(current)
        messages = [{"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": SHORTEN.format(least=least, most=most, reply=current)}]
        out = llm.complete(messages, tag=tag)["text"]
        if _words(out) >= _words(current):  # not shorter: one more try, like the rewrite below
            retry = messages + [{"role": "assistant", "content": out},
                                {"role": "user", "content": f"That is not shorter. Use at most {most} words. "
                                                            "Output only the reply."}]
            try:
                second = llm.complete(retry, tag=tag + ":retry")["text"]
            except llm.LLMError:
                second = ""
            if second and _words(second) < _words(out):
                out = second
        return generate.fill_phone(out, phone)
    messages = prompts.build_messages(business, review)
    extra = TONES.get(tone) or TONES[default_tone(int(review.get("stars") or 0))]
    if not _has_text(review):
        extra += "\n" + NO_TEXT
    messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n" + extra}
    best = llm.complete(messages, tag=tag)
    broken = generate.problems(best["text"], review, phone)
    if broken:
        fix = messages + [{"role": "assistant", "content": best["text"]},
                          {"role": "user", "content": "Rewrite it. Fix: " + "; ".join(broken)
                           + ". Output only the reply text."}]
        try:
            second = llm.complete(fix, tag=tag + ":fix")
        except llm.LLMError:
            second = None
        if second and len(generate.problems(second["text"], review, phone)) < len(broken):
            best = second
    return generate.fill_phone(best["text"], phone)
