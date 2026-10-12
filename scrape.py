"""Find local businesses and their newest Google reviews through Apify, then save them to the store.

Two actors, inputs checked against their default builds' input schemas:
  compass/crawler-google-places        discovery: free-text query -> places with phone, category, address
  compass/google-maps-reviews-scraper  newest reviews for the kept places, one run for all of them
The reviews actor can't do discovery alone: it takes no search text and its items carry no phone.
"""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

import config
import store

API = "https://api.apify.com/v2"
PLACES_ACTOR = "compass/crawler-google-places"
REVIEWS_ACTOR = "compass/google-maps-reviews-scraper"
RAW_DIR = config.DATA_DIR / "raw"
MIN_REVIEWS = 15
REVIEWS_PER_PLACE = 60
SYNC_TIMEOUT = 330  # the sync endpoint answers 408 after 300 s while the run keeps going
ACTIVE = ("READY", "RUNNING", "TIMING-OUT", "ABORTING")
PLACE_ID_RE = re.compile(r"(ChIJ|GhIJ)[A-Za-z0-9_-]{23}")  # the only ids the reviews actor accepts

LAST_RUN = {}  # counts from the latest scrape(), for print_table's summary line


# ---------- Apify ----------

def _headers():
    return {"Authorization": "Bearer " + config.require("APIFY_TOKEN")}


def _clean(text):
    """Error text that is safe to show: no token, short."""
    text = str(text)
    token = config.get("APIFY_TOKEN")
    if token:
        text = text.replace(token, "***")
    return re.sub(r"(?i)(token=)[^&\s]+", r"\1***", text)[:500]


def _error(resp):
    try:
        err = resp.json().get("error") or {}
        message = err.get("message") or err.get("type")
    except (ValueError, AttributeError):
        message = None
    return RuntimeError(_clean(f"Apify HTTP {resp.status_code}: {message or resp.text[:300]}"))


def _get(path, **params):
    try:
        resp = requests.get(API + path, headers=_headers(), params=params, timeout=120)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(_clean(f"Apify request failed: {exc}")) from None
    if not resp.ok:
        raise _error(resp)
    return resp.json()


def cache_path(actor_id, run_input):
    digest = hashlib.sha1(json.dumps(run_input, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return RAW_DIR / f"{actor_id.replace('/', '~')}_{digest}.json"


def apify_run(actor_id, run_input, use_cache=False):
    """Run an actor and return its dataset items; every result is cached in data/raw."""
    path = cache_path(actor_id, run_input)
    if use_cache:
        if not path.exists():
            raise RuntimeError(f"no cached result data/raw/{path.name}; run once without --from-cache")
        return json.loads(path.read_text(encoding="utf-8"))
    act = actor_id.replace("/", "~")
    started = datetime.now(timezone.utc)
    try:
        resp = requests.post(f"{API}/acts/{act}/run-sync-get-dataset-items", headers=_headers(),
                             params={"clean": "true", "format": "json"}, json=run_input,
                             timeout=SYNC_TIMEOUT)
    except requests.exceptions.ReadTimeout:
        resp = None
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(_clean(f"Apify request failed: {exc}")) from None
    if resp is None or resp.status_code == 408:
        items = _finish_inflight_run(act, started, run_input)
    elif resp.ok:
        items = resp.json()
    else:
        raise _error(resp)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


def _same_input(run, run_input):
    """Apify stores the posted input plus schema defaults, so ours must be contained in it."""
    stored = _get(f"/key-value-stores/{run['defaultKeyValueStoreId']}/records/INPUT")
    return isinstance(stored, dict) and all(stored.get(k) == v for k, v in run_input.items())


def _finish_inflight_run(act, started, run_input):
    """The sync call gave up but the run goes on: wait for that run instead of starting another."""
    runs = _get(f"/acts/{act}/runs", desc=1, limit=5)["data"]["items"]
    # another scrape may be running this actor right now, so only a run with our input counts
    run = next((r for r in runs if r.get("status") in ("READY", "RUNNING")
                and _same_input(r, run_input)), None)
    if run is None:  # it may have finished between the 408 and this lookup
        since = started - timedelta(minutes=2)
        run = next((r for r in runs if (store.parse_iso(r.get("startedAt")) or since) > since
                    and _same_input(r, run_input)), None)
    if run is None:
        raise RuntimeError(f"Apify sync call for {act} timed out and its run could not be found")
    print(f"  Apify is still working; waiting for run {run['id']} ...", flush=True)
    for _ in range(40):  # waitForFinish=60 -> up to ~40 minutes
        run = _get(f"/actor-runs/{run['id']}", waitForFinish=60)["data"]
        if run["status"] not in ACTIVE:
            break
    else:
        raise RuntimeError(f"Apify run {run['id']} is still running; check the Apify console")
    if run["status"] != "SUCCEEDED":
        raise RuntimeError(f"Apify run {run['id']} ended with status {run['status']}")
    return _get(f"/datasets/{run['defaultDatasetId']}/items", clean="true", format="json")


def places_input(query, max_places):
    """Discovery: search results only, closed places skipped, every other paid add-on off."""
    return {
        "searchStringsArray": [query],
        "maxCrawledPlacesPerSearch": max_places,
        "language": "en",
        # billed as "filter-applied" (~$0.001/place); scrape() also drops closed places itself
        "skipClosedPlaces": True,
        "scrapePlaceDetailPage": False,
        "maxReviews": 0,
        "maxImages": 0,
        "maxQuestions": 0,
        "scrapeImageAuthors": False,
        "scrapeContacts": False,
        "scrapeSocialMediaProfiles": {"facebooks": False, "instagrams": False, "youtubes": False,
                                      "tiktoks": False, "twitters": False},
        "maximumLeadsEnrichmentRecords": 0,
        "verifyLeadsEnrichmentEmails": False,
        "scrapeTableReservationProvider": False,
        "scrapeOrderOnline": False,
        "includeWebResults": False,
        "scrapeDirectories": False,
        "enableCompetitorAnalysis": False,
        "maxCompetitorsToAnalyze": 0,
    }


def reviews_input(places):
    """One run for all kept places: newest Google reviews, reviewer names on."""
    run_input = {
        "maxReviews": REVIEWS_PER_PLACE,
        "reviewsSort": "newest",
        "language": "en",
        "reviewsOrigin": "google",
        "personalData": True,
    }
    ids = [p["place_id"] for p in places if PLACE_ID_RE.fullmatch(p["place_id"])]
    urls = [{"url": p["maps_url"]} for p in places
            if not PLACE_ID_RE.fullmatch(p["place_id"]) and p["maps_url"]]
    if ids:
        run_input["placeIds"] = ids
    if urls:
        run_input["startUrls"] = urls
    return run_input


# ---------- normalizing ----------

def iso_utc(value):
    """'2025-01-15T10:30:00.000Z' -> '2025-01-15T10:30:00+00:00'; None when missing or unparseable."""
    dt = store.parse_iso(value)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else None


def slugify(name, key):
    """ASCII slug of the name (max 40 chars) + 4 hex chars of sha1(key), key = place_id or maps_url."""
    base = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii").lower()
    base = base.replace("'", "")  # Booker's -> bookers, same as the curly apostrophe the fold drops
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")[:40].strip("-") or "place"
    return base + "-" + hashlib.sha1((key or "").encode("utf-8")).hexdigest()[:4]


def normalize_place(item):
    """Places-actor item -> business doc (review_count may be None; unanswered_count added later)."""
    place_id = item.get("placeId") or ""
    maps_url = item.get("url") or ""
    count = item.get("reviewsCount")
    return {
        "_id": slugify(item.get("title"), place_id or maps_url),
        "name": (item.get("title") or "").strip(),
        "category": item.get("categoryName") or "",
        "address": item.get("address") or "",
        "phone": item.get("phone") or item.get("phoneUnformatted") or "",
        "maps_url": maps_url,
        "rating": item.get("totalScore"),
        "review_count": int(count) if count is not None else None,
        "place_id": place_id,
    }


def normalize_review(item, business_id):
    """Reviews-actor item -> review doc."""
    author = (item.get("name") or "").strip()
    text = (item.get("text") or "").strip()
    date = iso_utc(item.get("publishedAtDate"))
    review_id = item.get("reviewId") or hashlib.sha1(
        f"{item.get('placeId') or ''}|{author}|{date or ''}|{text}".encode("utf-8")).hexdigest()
    return {
        "_id": review_id,
        "business_id": business_id,
        "author": author,
        "stars": int(item.get("stars") or 0),
        "text": text,
        "date": date,
        "owner_replied": bool(item.get("responseFromOwnerText") or item.get("responseFromOwnerDate")),
    }


# ---------- pipeline ----------

def scrape(query, max_places=10, use_cache=False):
    """Discover places, fetch their newest reviews, save businesses with >= 15 reviews. Returns them."""
    store.get_business("")  # the backend connects lazily: fail here, before any paid Apify run
    print(f"Searching Google Maps: {query!r} (max {max_places} places) ...", flush=True)
    places, seen = [], set()
    for item in apify_run(PLACES_ACTOR, places_input(query, max_places), use_cache):
        if not item.get("title") or item.get("permanentlyClosed") or item.get("temporarilyClosed"):
            continue
        place = normalize_place(item)
        if place["_id"] not in seen:
            seen.add(place["_id"])
            places.append(place)

    # Google's total picks who gets a reviews run; an unknown total is checked after scraping.
    candidates = [p for p in places if p["review_count"] is None or p["review_count"] >= MIN_REVIEWS]
    reviews = {}
    if candidates:
        print(f"Fetching up to {REVIEWS_PER_PLACE} newest reviews for {len(candidates)} places ...", flush=True)
        by_id = {p["place_id"]: p["_id"] for p in candidates if p["place_id"]}
        by_url = {p["maps_url"]: p["_id"] for p in candidates if p["maps_url"]}
        for item in apify_run(REVIEWS_ACTOR, reviews_input(candidates), use_cache):
            slug = (by_id.get(item.get("placeId")) or by_id.get(item.get("inputPlaceId"))
                    or by_url.get(item.get("url")) or by_url.get(item.get("inputStartUrl")))
            if slug:
                review = normalize_review(item, slug)
                reviews[review["_id"]] = review

    kept, kept_reviews = [], []
    for place in candidates:
        mine = [r for r in reviews.values() if r["business_id"] == place["_id"]]
        if place["review_count"] is None:
            place["review_count"] = len(mine)
        if place["review_count"] < MIN_REVIEWS:
            continue
        place["unanswered_count"] = sum(1 for r in mine if not r["owner_replied"])
        kept.append(place)
        kept_reviews.extend(mine)

    reviews_saved = store.upsert_reviews(kept_reviews) if kept_reviews else 0
    saved = [store.upsert_business(place) for place in kept]
    LAST_RUN.clear()
    LAST_RUN.update(found=len(places), kept=len(saved), reviews_saved=reviews_saved)
    return saved


def print_table(businesses, summary=None):
    """Businesses sorted by unanswered reviews, then one summary line (scrape counts if given)."""
    rows = sorted(businesses, key=lambda b: (-(b.get("unanswered_count") or 0), b.get("name") or ""))
    print(f"{'name':34}  {'phone':16}  {'reviews':>7}  {'unanswered':>10}  slug")
    print("-" * 100)
    total_reviews = total_unanswered = 0
    for b in rows:
        scraped = len(store.reviews_for(b["_id"]))
        unanswered = b.get("unanswered_count") or 0
        total_reviews += scraped
        total_unanswered += unanswered
        name = (b.get("name") or "")[:34]
        print(f"{name:34}  {(b.get('phone') or '-'):16}  {scraped:>7}  {unanswered:>10}  {b['_id']}")
    if summary:
        print(f"places found: {summary.get('found', 0)} | kept: {summary.get('kept', len(rows))} | "
              f"reviews saved: {summary.get('reviews_saved', 0)} | unanswered: {total_unanswered} | "
              f"store: {store.BACKEND}")
    else:
        print(f"businesses: {len(rows)} | reviews: {total_reviews} | unanswered: {total_unanswered} | "
              f"store: {store.BACKEND}")
