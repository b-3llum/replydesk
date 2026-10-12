"""Reply quality for the Reply Studio: the deterministic rubric from evolve/score.py as a 0-100 score.

    quality.report(review, business, reply) -> {"score", "bars", "checks"}

No LLM call: only the 65 deterministic points are used, never evolve's LLM judge. evolve/ belongs to
the prompt-evolution work, so its checks are copied here instead of imported.
"""
import re

# BANNED through POINTS mirror evolve/score.py exactly (only checks() takes the business as an argument).
BANNED = ("valued", "feedback", "strive", "apologies for any inconvenience", "sorry you feel")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u200D]")
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

# CONTENT WORDS: lowercase alphabetic tokens of length >= 4 that are not stopwords, not generic
# review/reply vocabulary, and not words from the business's own name or category.
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


def checks(review, business, reply):
    """{check: passed} for the 65 deterministic points (evolve's deterministic_checks)."""
    biz = business or {}
    reply = reply or ""
    words = re.findall("[A-Za-z0-9’']+", reply)
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
        "specific": bool(content_words(reply, exclude) & content_words(review.get("text"), exclude)),
        "name": name is None or re.search(rf"\b{re.escape(name)}\b", reply, re.I) is not None,
        "outcome": outcome,
        "clean": (not any(b in lowered for b in BANNED) and not EMOJI.search(reply)
                  and reply.count("!") <= 1),
    }


POINTS = {"length": 15, "specific": 15, "name": 10, "outcome": 15, "clean": 10}

# ---------------------------------------------------------------- studio mapping

# (key, label, checks that feed it): each bar is the share of its checks' points, 0-100.
BARS = (
    ("personalization", "Personalization", ("specific", "name")),
    ("professionalism", "Professionalism", ("length",)),
    ("complaint", "Complaint addressed", ("outcome",)),
    ("policy", "Policy safety", ("clean",)),
)


def _percent(passed, keys):
    return round(sum(POINTS[k] for k in keys if passed[k]) / sum(POINTS[k] for k in keys) * 100)


def report(review, business, reply):
    """QUALITY: score = points earned / 65 as 0-100, four 0-100 bars, and the raw checks."""
    passed = checks(review, business, reply)
    return {
        "score": _percent(passed, POINTS),
        "bars": [{"key": key, "label": label, "value": _percent(passed, keys)} for key, label, keys in BARS],
        "checks": passed,
    }
