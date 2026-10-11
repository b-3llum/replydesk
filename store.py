"""ReplyDesk data layer: one interface, two backends (JSON file or MongoDB).

    import store
    store.upsert_business({"_id": "joes-pizza", "name": "Joe's Pizza", ...})

REPLYDESK_STORE=json|mongo picks the backend; empty means mongo when MONGODB_URI is set,
else json. The backend is created on first use, so importing this module never blocks.
All datetimes are ISO-8601 UTC strings, so both backends hold identical documents.
"""
import contextlib
import copy
import json
import os
import re
import secrets
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import certifi
from pymongo import MongoClient, ReplaceOne, ReturnDocument, UpdateOne

import config

# Set when a business is first inserted. upsert_business never changes them afterwards
# (mark_paid / mark_texted do), so a re-scrape can't reset a paying customer.
BUSINESS_LIFECYCLE = ("preview_token", "paid", "status", "created_at")
# Owned by save_reply. upsert_reviews sets them on insert only.
REVIEW_DEFAULTS = {"reply": None, "model": None, "tokens_in": 0, "tokens_out": 0, "generated_at": None}
READY_QUERY = {"owner_replied": False, "reply": {"$ne": None}}
SECRET_KEYS = ("MONGODB_URI", "APIFY_TOKEN", "LUMINAL_API_KEY", "ADMIN_KEY", "PAYMENT_LINK")


def _pick_backend():
    mode = (config.get("REPLYDESK_STORE") or "").lower()
    if mode in ("json", "mongo"):
        return mode
    if mode not in ("", "auto"):
        raise RuntimeError(f"REPLYDESK_STORE must be json, mongo or empty, not {mode!r}")
    return "mongo" if config.get("MONGODB_URI") else "json"


def _json_path():
    raw = config.get("REPLYDESK_JSON_PATH")
    path = Path(raw) if raw else config.DATA_DIR / "store.json"
    return path if path.is_absolute() else config.BASE_DIR / path


BACKEND = _pick_backend()
JSON_PATH = _json_path()
MONGO_DB = config.get("REPLYDESK_MONGO_DB") or "replydesk"


# ---------------------------------------------------------------- shared helpers

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_ready(review):
    return review.get("owner_replied") is False and review.get("reply") is not None


def _is_pending(review):
    return review.get("owner_replied") is False and review.get("reply") is None


def _sort_businesses(docs):
    return sorted(docs, key=lambda b: (-_int(b.get("unanswered_count")),
                                       str(b.get("name") or "").lower(), str(b["_id"])))


def _sort_reviews(docs):
    """Newest date first, missing dates last."""
    dated = sorted((r for r in docs if r.get("date")),
                   key=lambda r: (str(r["date"]), str(r["_id"])), reverse=True)
    undated = sorted((r for r in docs if not r.get("date")), key=lambda r: str(r["_id"]))
    return dated + undated


def _sort_orders(docs):
    """Newest paid_at first. Input is newest-inserted first, and the stable sort keeps that for ties."""
    return sorted(docs, key=lambda o: str(o.get("paid_at") or ""), reverse=True)


def _local_date(iso):
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().date()


def _order_stats(orders):
    today = date.today()
    revenue = revenue_today = orders_today = 0
    for order in orders:
        amount = order.get("amount") or 0
        revenue += amount
        if _local_date(order.get("paid_at")) == today:
            orders_today += 1
            revenue_today += amount
    return {"paid_orders": len(orders), "revenue": revenue,
            "orders_today": orders_today, "revenue_today": revenue_today}


def _business_input(doc):
    doc = copy.deepcopy(dict(doc))
    if not doc.get("_id"):
        raise ValueError("business doc needs an _id (the slug)")
    doc["_id"] = str(doc["_id"])
    return doc


def _business_insert_fields(doc):
    """Lifecycle fields for a new business: the caller's value when given, else the default."""
    defaults = {"preview_token": secrets.token_urlsafe(8), "paid": False,
                "status": "new", "created_at": _now()}
    return {k: doc[k] if doc.get(k) is not None else v for k, v in defaults.items()}


def _business_update_fields(doc):
    return {k: v for k, v in doc.items() if k != "_id" and k not in BUSINESS_LIFECYCLE}


def _review_input(review):
    review = copy.deepcopy(dict(review))
    if not review.get("_id"):
        raise ValueError("review needs an _id")
    review["_id"] = str(review["_id"])
    if "owner_replied" in review:
        review["owner_replied"] = bool(review["owner_replied"])
    return review


def _scraped_fields(review):
    """What a (re-)scrape may write: everything except _id and the reply fields."""
    return {k: v for k, v in review.items() if k != "_id" and k not in REVIEW_DEFAULTS}


def _reply_fields(reply, model, tokens_in, tokens_out):
    return {"reply": reply, "model": model, "tokens_in": _int(tokens_in),
            "tokens_out": _int(tokens_out), "generated_at": _now()}


# ---------------------------------------------------------------- JSON backend

def _empty():
    return {"businesses": {}, "reviews": {}, "orders": [], "requests": []}


def _parse(raw):
    data = json.loads(raw.decode("utf-8-sig")) if raw.strip() else _empty()
    if not isinstance(data, dict):
        raise ValueError("store file is not a JSON object")
    for key, kind in (("businesses", dict), ("reviews", dict), ("orders", list), ("requests", list)):
        if not isinstance(data.setdefault(key, kind()), kind):
            raise ValueError(f"store file: {key!r} has the wrong type")
    return data


class JsonStore:
    """JSON file store, safe across threads (one RLock) and processes (lock file + atomic replace).

    File layout: {"businesses": {slug: doc}, "reviews": {review_id: doc}, "orders": [doc, ...]}
    """

    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        self.lock_path = self.path + ".lock"
        self._lock = threading.RLock()
        self._released_at = self._burst_since = 0.0  # for the fairness gap in _file_lock
        self._data = _empty()
        # (st_ino, st_mtime_ns, st_size) of the file self._data came from. st_ino matters:
        # two same-size writes can land in one mtime tick on Windows, but each replace is a new file.
        self._sig = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock, self._file_lock():
            if os.path.exists(self.path):
                self._refresh(strict=True)
            else:
                self._write()

    # -- plumbing

    def _refresh(self, strict=False):
        """Reload from disk if the file changed since the last load.

        Reads keep the last good copy when the file can't be read right now; mutations
        (strict) raise instead, so they never write over data they could not read.
        """
        error = None
        for _ in range(25 if strict else 5):
            try:
                st = os.stat(self.path)
            except FileNotFoundError:
                self._data, self._sig = _empty(), None  # deleted: start over empty
                return
            except OSError as e:
                error = e
                time.sleep(0.02)
                continue
            sig = (st.st_ino, st.st_mtime_ns, st.st_size)
            if sig == self._sig and self._data is not None:
                return
            try:
                # sig was taken before the read, so it is never newer than the data:
                # a write in between only causes one extra reload later.
                with open(self.path, "rb") as f:
                    raw = f.read()
                self._data, self._sig = _parse(raw), sig
                return
            except (OSError, ValueError) as e:  # PermissionError mid-replace, JSONDecodeError
                error = e
                time.sleep(0.02)
        if strict or self._data is None:
            raise RuntimeError(f"could not read {self.path}: {type(error).__name__}: {error}")

    @contextlib.contextmanager
    def _file_lock(self):
        """Cross-process mutex: an O_EXCL lock file next to the store."""
        # Fairness: back-to-back writes (e.g. 40 reply threads) re-take the lock within
        # microseconds, so a waiter polling every 20 ms could starve past its 15 s timeout.
        # After a second of continuous use, leave a gap long enough for one poll to land.
        now = time.monotonic()
        if now - self._released_at > 0.05:
            self._burst_since = now
        elif now - self._burst_since > 1.0:
            time.sleep(0.05)
            self._burst_since = time.monotonic()
        deadline = time.monotonic() + 15
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):
                pass  # held elsewhere (PermissionError: Windows is still deleting it)
            try:
                if time.time() - os.stat(self.lock_path).st_mtime > 30:
                    os.remove(self.lock_path)  # stale: its owner died
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"store lock still busy after 15s: {self.lock_path}")
            time.sleep(0.02)
        try:
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            yield
        finally:
            for _ in range(50):
                try:
                    os.remove(self.lock_path)
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    time.sleep(0.02)
            self._released_at = time.monotonic()

    def _write(self):
        """Atomic write: temp file in the same dir, fsync, os.replace (retried on Windows)."""
        text = json.dumps(self._data, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path),
                                   prefix=os.path.basename(self.path) + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(text.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            st = os.stat(tmp)  # a rename keeps ino/mtime/size, so this is the new file's sig
            for attempt in range(50):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:  # another process has the store open right now
                    if attempt == 49:
                        raise
                    time.sleep(0.02)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        self._sig = (st.st_ino, st.st_mtime_ns, st.st_size)

    def _mutate(self, fn):
        """fn(data) -> (result, changed), run on fresh data under both locks; writes if changed."""
        with self._lock, self._file_lock():
            self._refresh(strict=True)
            try:
                result, changed = fn(self._data)
                if changed:
                    self._write()
            except BaseException:
                self._data, self._sig = None, None  # maybe half-applied: reload from disk next time
                raise
            return result

    def _read(self, fn):
        with self._lock:
            self._refresh()
            return copy.deepcopy(fn(self._data))

    # -- API

    def upsert_business(self, doc):
        doc = _business_input(doc)

        def apply(data):
            current = data["businesses"].get(doc["_id"])
            if current is None:
                current = dict(doc, **_business_insert_fields(doc))
                data["businesses"][doc["_id"]] = current
            else:
                current.update(_business_update_fields(doc))
            return copy.deepcopy(current), True

        return self._mutate(apply)

    def get_business(self, slug):
        def get(data):
            biz = data["businesses"].get(slug)
            if biz is None:
                return None
            ready = sum(1 for r in data["reviews"].values()
                        if r.get("business_id") == slug and _is_ready(r))
            return dict(biz, replies_ready=ready)

        return self._read(get)

    def list_businesses(self):
        def get(data):
            ready = Counter(r.get("business_id") for r in data["reviews"].values() if _is_ready(r))
            return _sort_businesses([dict(b, replies_ready=ready[b["_id"]])
                                     for b in data["businesses"].values()])

        return self._read(get)

    def upsert_reviews(self, reviews):
        reviews = [_review_input(r) for r in reviews]
        if not reviews:
            return 0

        def apply(data):
            for review in reviews:
                current = data["reviews"].get(review["_id"])
                if current is None:
                    data["reviews"][review["_id"]] = {"_id": review["_id"], **_scraped_fields(review),
                                                      **REVIEW_DEFAULTS}
                else:
                    current.update(_scraped_fields(review))
            return len(reviews), True

        return self._mutate(apply)

    def reviews_for(self, slug):
        return self._read(lambda data: _sort_reviews(
            [r for r in data["reviews"].values() if r.get("business_id") == slug]))

    def pending_reviews(self, slug=None):
        return self._read(lambda data: _sort_reviews(
            [r for r in data["reviews"].values()
             if _is_pending(r) and (slug is None or r.get("business_id") == slug)]))

    def save_reply(self, review_id, reply, model, tokens_in, tokens_out):
        fields = _reply_fields(reply, model, tokens_in, tokens_out)

        def apply(data):
            review = data["reviews"].get(review_id)
            if review is None:
                raise KeyError(f"unknown review {review_id!r}")
            review.update(fields)
            return None, True

        self._mutate(apply)

    def mark_paid(self, slug, method="stripe", amount=config.PRICE_USD):
        order = {"_id": secrets.token_hex(12), "business_id": slug, "amount": amount,
                 "method": method, "paid_at": _now()}

        def apply(data):
            biz = data["businesses"].get(slug)
            if biz is None or biz.get("paid") is True:
                return False, False
            biz["paid"], biz["status"] = True, "paid"
            data["orders"].append(order)
            return True, True

        return self._mutate(apply)

    def mark_texted(self, slug):
        def apply(data):
            biz = data["businesses"].get(slug)
            if biz is None or biz.get("paid") is True or biz.get("status") == "paid":
                return False, False
            changed = biz.get("status") != "texted"
            biz["status"] = "texted"
            return True, changed

        return self._mutate(apply)

    def list_orders(self):
        return self._read(lambda data: _sort_orders(list(reversed(data["orders"]))))

    def add_request(self, doc):
        doc = dict(doc, _id=secrets.token_hex(12))

        def apply(data):
            data["requests"].append(doc)
            return dict(doc), True

        return self._mutate(apply)

    def list_requests(self, limit=50):
        return self._read(lambda data: [dict(r) for r in reversed(data["requests"])][:limit])

    def replies_since(self, iso):
        def count(data):
            rows = [r for r in data["reviews"].values() if (r.get("generated_at") or "") >= iso
                    and r.get("model") not in (None, "template")]
            return {"replies": len(rows), "tokens_in": sum(r.get("tokens_in") or 0 for r in rows),
                    "tokens_out": sum(r.get("tokens_out") or 0 for r in rows)}

        return self._read(count)

    def stats(self):
        def get(data):
            reviews = list(data["reviews"].values())
            out = {
                "businesses": len(data["businesses"]),
                "reviews": len(reviews),
                "unanswered": sum(1 for r in reviews if r.get("owner_replied") is False),
                "replies_ready": sum(1 for r in reviews if _is_ready(r)),
            }
            out.update(_order_stats(data["orders"]))
            out["tokens_in"] = sum(_int(r.get("tokens_in")) for r in reviews)
            out["tokens_out"] = sum(_int(r.get("tokens_out")) for r in reviews)
            out["llm_calls"] = sum(1 for r in reviews if r.get("model") not in (None, "template"))
            return out

        return self._read(get)

    def export_all(self):
        """Copies of every document, for migration and backups."""
        return self._read(lambda data: {"businesses": list(data["businesses"].values()),
                                        "reviews": list(data["reviews"].values()),
                                        "orders": list(data["orders"])})

    def close(self):
        pass


# ---------------------------------------------------------------- Mongo backend

class MongoStore:
    """MongoDB store: collections businesses / reviews / orders in db_name."""

    def __init__(self, uri, db_name="replydesk"):
        uri = _escape_credentials(uri)
        client = None
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=8000, tlsCAFile=certifi.where())
            client.admin.command("ping")
        except Exception as e:
            if client is not None:
                client.close()
            raise RuntimeError(f"MongoDB unreachable ({_short_reason(e, uri)}). Check Atlas Network "
                               "Access, or set REPLYDESK_STORE=json in .env") from None
        self.client = client
        self.db = client[db_name]
        self.db.reviews.create_index("business_id")
        self.db.reviews.create_index("reply")

    def upsert_business(self, doc):
        doc = _business_input(doc)
        update = {"$setOnInsert": _business_insert_fields(doc)}
        fields = _business_update_fields(doc)
        if fields:
            update["$set"] = fields
        return self.db.businesses.find_one_and_update(
            {"_id": doc["_id"]}, update, upsert=True, return_document=ReturnDocument.AFTER)

    def get_business(self, slug):
        biz = self.db.businesses.find_one({"_id": slug})
        if biz is not None:
            biz["replies_ready"] = self.db.reviews.count_documents(dict(READY_QUERY, business_id=slug))
        return biz

    def list_businesses(self):
        ready = {row["_id"]: row["n"] for row in self.db.reviews.aggregate(
            [{"$match": READY_QUERY}, {"$group": {"_id": "$business_id", "n": {"$sum": 1}}}])}
        docs = list(self.db.businesses.find())
        for biz in docs:
            biz["replies_ready"] = ready.get(biz["_id"], 0)
        return _sort_businesses(docs)

    def upsert_reviews(self, reviews):
        reviews = [_review_input(r) for r in reviews]
        if not reviews:
            return 0
        ops = []
        for review in reviews:
            update = {"$setOnInsert": dict(REVIEW_DEFAULTS)}
            fields = _scraped_fields(review)
            if fields:
                update["$set"] = fields
            ops.append(UpdateOne({"_id": review["_id"]}, update, upsert=True))
        self.db.reviews.bulk_write(ops)
        return len(ops)

    def reviews_for(self, slug):
        return _sort_reviews(list(self.db.reviews.find({"business_id": slug})))

    def pending_reviews(self, slug=None):
        query = {"owner_replied": False, "reply": None}
        if slug is not None:
            query["business_id"] = slug
        return _sort_reviews(list(self.db.reviews.find(query)))

    def save_reply(self, review_id, reply, model, tokens_in, tokens_out):
        result = self.db.reviews.update_one(
            {"_id": review_id}, {"$set": _reply_fields(reply, model, tokens_in, tokens_out)})
        if result.matched_count == 0:
            raise KeyError(f"unknown review {review_id!r}")

    def mark_paid(self, slug, method="stripe", amount=config.PRICE_USD):
        # Only one caller can flip paid, so double clicks can't create two orders.
        result = self.db.businesses.update_one(
            {"_id": slug, "paid": {"$ne": True}}, {"$set": {"paid": True, "status": "paid"}})
        if result.matched_count == 0:
            return False
        self.db.orders.insert_one({"business_id": slug, "amount": amount, "method": method,
                                   "paid_at": _now()})
        return True

    def mark_texted(self, slug):
        result = self.db.businesses.update_one(
            {"_id": slug, "paid": {"$ne": True}, "status": {"$ne": "paid"}},
            {"$set": {"status": "texted"}})
        return result.matched_count == 1

    def list_orders(self):
        orders = list(self.db.orders.find().sort([("paid_at", -1), ("_id", -1)]))
        for order in orders:
            order["_id"] = str(order["_id"])
        return _sort_orders(orders)

    def add_request(self, doc):
        doc = dict(doc)
        self.db.requests.insert_one(doc)
        doc["_id"] = str(doc["_id"])
        return doc

    def list_requests(self, limit=50):
        rows = list(self.db.requests.find().sort([("created_at", -1), ("_id", -1)]).limit(limit))
        for row in rows:
            row["_id"] = str(row["_id"])
        return rows

    def replies_since(self, iso):
        rows = list(self.db.reviews.aggregate([
            {"$match": {"generated_at": {"$gte": iso}, "model": {"$nin": [None, "template"]}}},
            {"$group": {"_id": None, "n": {"$sum": 1}, "tin": {"$sum": "$tokens_in"}, "tout": {"$sum": "$tokens_out"}}}]))
        row = rows[0] if rows else {}
        return {"replies": row.get("n", 0), "tokens_in": row.get("tin", 0), "tokens_out": row.get("tout", 0)}

    def stats(self):
        reviews = self.db.reviews
        tokens = list(reviews.aggregate([{"$group": {
            "_id": None, "tin": {"$sum": "$tokens_in"}, "tout": {"$sum": "$tokens_out"}}}]))
        out = {
            "businesses": self.db.businesses.count_documents({}),
            "reviews": reviews.count_documents({}),
            "unanswered": reviews.count_documents({"owner_replied": False}),
            "replies_ready": reviews.count_documents(READY_QUERY),
        }
        out.update(_order_stats(list(self.db.orders.find())))
        out["tokens_in"] = tokens[0]["tin"] if tokens else 0
        out["tokens_out"] = tokens[0]["tout"] if tokens else 0
        out["llm_calls"] = reviews.count_documents({"model": {"$nin": [None, "template"]}})
        return out

    def close(self):
        self.client.close()


# ---------------------------------------------------------------- module API

_backend = None
_backend_lock = threading.Lock()
_backend_error = None  # (monotonic time, message): fail fast for 10s instead of re-dialing per call


def _store():
    global _backend, _backend_error
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is None:
            if _backend_error and time.monotonic() - _backend_error[0] < 10:
                raise RuntimeError(_backend_error[1])
            try:
                if BACKEND == "mongo":
                    _backend = MongoStore(config.require("MONGODB_URI"), MONGO_DB)
                else:
                    _backend = JsonStore(JSON_PATH)
            except Exception as e:
                _backend_error = (time.monotonic(), str(e))
                raise
        return _backend


def upsert_business(doc):
    """Insert or merge a business by _id (slug). Lifecycle fields are only set on insert."""
    return _store().upsert_business(doc)


def get_business(slug):
    return _store().get_business(slug)


def list_businesses():
    return _store().list_businesses()


def upsert_reviews(reviews):
    """Insert or update scraped reviews; never touches reply/model/tokens/generated_at."""
    return _store().upsert_reviews(reviews)


def reviews_for(slug):
    return _store().reviews_for(slug)


def pending_reviews(slug=None):
    return _store().pending_reviews(slug)


def save_reply(review_id, reply, model, tokens_in, tokens_out):
    _store().save_reply(review_id, reply, model, tokens_in, tokens_out)


def mark_paid(slug, method="stripe", amount=config.PRICE_USD):
    """True if this call marked it paid (and recorded the order); False if already paid or unknown."""
    return _store().mark_paid(slug, method, amount)


def mark_texted(slug):
    """True if status is now texted; False if the business is paid or unknown."""
    return _store().mark_texted(slug)


def list_orders():
    return _store().list_orders()


def add_request(name, phone, maps_url="", plan="one-time"):
    """Save a landing-page lead to "requests": {name, phone, maps_url, plan, created_at, status: new}."""
    return _store().add_request({"name": name, "phone": phone, "maps_url": maps_url, "plan": plan,
                                 "created_at": _now(), "status": "new"})


def list_requests(limit=50):
    """Landing-page leads, newest first."""
    return _store().list_requests(limit)


def replies_since(iso):
    """LLM replies written at or after iso (UTC ISO string): {replies, tokens_in, tokens_out}."""
    return _store().replies_since(iso)


def find_businesses(query, limit=8):
    """Case-insensitive partial match on business name, most unanswered first."""
    query = " ".join((query or "").lower().split())
    if not query:
        return []
    return [b for b in list_businesses() if query in " ".join((b.get("name") or "").lower().split())][:limit]


def stats():
    return _store().stats()


def migrate_json_to_mongo(json_path=None, uri=None, db_name=None):
    """Copy the JSON store into Mongo (JSON wins; orders matched on business_id + paid_at). Idempotent."""
    json_path = os.fspath(json_path or JSON_PATH)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"no JSON store at {json_path}")
    data = JsonStore(json_path).export_all()
    mongo = MongoStore(uri or config.require("MONGODB_URI"), db_name or MONGO_DB)
    inserted = skipped = 0
    try:
        db = mongo.db
        if data["businesses"]:
            db.businesses.bulk_write([ReplaceOne({"_id": b["_id"]}, b, upsert=True)
                                      for b in data["businesses"]])
        if data["reviews"]:
            db.reviews.bulk_write([ReplaceOne({"_id": r["_id"]}, r, upsert=True)
                                   for r in data["reviews"]])
        for order in data["orders"]:
            order = {k: v for k, v in order.items() if k != "_id"}  # Mongo assigns its own ObjectId
            if db.orders.find_one({"business_id": order.get("business_id"),
                                   "paid_at": order.get("paid_at")}):
                skipped += 1
            else:
                db.orders.insert_one(order)
                inserted += 1
    finally:
        mongo.close()
    return {"businesses": len(data["businesses"]), "reviews": len(data["reviews"]),
            "orders_inserted": inserted, "orders_skipped": skipped}


# ---------------------------------------------------------------- secret hygiene

def redact(text, uri=None):
    """text with secret config values (and Mongo user, password and hosts) replaced by ***."""
    text = re.sub(r"mongodb(?:\+srv)?://\S+", "mongodb://***", str(text))
    words = [config.get(name) for name in SECRET_KEYS]
    for u in {uri, config.get("MONGODB_URI")}:
        uri_words, domains = _uri_parts(u)
        words += uri_words
        for domain in domains:  # shard hosts share the cluster's domain
            text = re.sub(r"[\w.-]*" + re.escape(domain), "***", text, flags=re.I)
    for word in sorted({w for w in words if w and len(w) >= 3}, key=len, reverse=True):
        text = text.replace(word, "***")
    return text


def _escape_credentials(uri):
    """Percent-encode user and password, so a raw Atlas password with @ : / ? # works as pasted."""
    if not uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    userinfo, sep, hostpart = rest.rpartition("@")  # hosts never contain @
    if not sep or re.search(r"%[0-9A-Fa-f]{2}", userinfo):  # no credentials, or already encoded
        return uri
    user, colon, password = userinfo.partition(":")
    quote = urllib.parse.quote_plus
    return f"{scheme}://{quote(user)}{colon}{quote(password)}@{hostpart}"


def _uri_parts(uri):
    """([user, password, hosts...], [domain suffixes]) of a Mongo URI."""
    if not uri or "://" not in uri:
        return [], []
    userinfo, _, hostpart = uri.split("://", 1)[1].rpartition("@")  # hosts never contain @
    words = []
    if userinfo:
        user, _, password = userinfo.partition(":")
        words += [userinfo, user, password, urllib.parse.unquote(user), urllib.parse.unquote(password)]
        # pymongo's parse errors can echo a piece of an unescaped password (e.g. "port must be...")
        words += re.split(r"[/?#@:\[\]]", userinfo)
    domains = []
    for host in re.split(r"[/?]", hostpart, maxsplit=1)[0].split(","):
        host = host.split(":")[0]
        words.append(host)
        labels = host.split(".")
        if len(labels) >= 3:
            domains.append(".".join(labels[1:]))
    return words, domains


def _short_reason(exc, uri=None):
    msg = str(exc)
    for cut in (", full error:", ", Timeout:", "Topology Description:"):  # drop pymongo's dumps
        msg = msg.split(cut)[0]
    msg = " ".join(redact(msg, uri).split())
    if len(msg) > 200:
        msg = msg[:200] + "..."
    if getattr(exc, "code", None) == 18:  # the server answered, so it's the login, not the network
        msg += " - check the user and password in MONGODB_URI (Atlas > Database Access)"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
