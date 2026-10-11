"""ReplyDesk web app: the owner's preview page, the paid full set, and the Stripe return page.

    python run.py serve          (uvicorn on 0.0.0.0:8000)

    GET /p/{slug}?t=token        preview: 5 replies readable, the rest locked (real text never sent)
    GET /full/{slug}?t=token     every reply, once the business has paid
    GET /thanks                  where Stripe sends the owner after paying
"""
import secrets
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
import evolution
import llm
import outreach
import store

SHOWCASE_STARS = (1, 2, 3, 5, 4)  # the preview opens with one written review of each, in this order
FREE_CARDS = 5  # readable cards on the preview; every card after these is locked
MAX_CARDS = 200  # cap per page, however many reviews pile up over re-scrapes
LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua").split()


@asynccontextmanager
async def lifespan(app):
    # The first MongoDB connection takes ~5 s: pay it at startup, not on the owner's first view.
    try:
        store.list_businesses()
    except Exception as exc:
        print(f"warning: store warm-up failed, pages will retry: {exc}")
    yield


app = FastAPI(title="ReplyDesk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "templates")

FRIENDLY = {
    403: ("This link isn't valid", "It may be incomplete or out of date. Open the link from the text we "
                                   "sent you, or reply to that text and we'll send a fresh one."),
    404: ("Page not found", "We couldn't find that page. Check the link in your text message."),
}


def _wants_json(request):
    return request.url.path.endswith(".json") or request.url.path == "/health"


@app.exception_handler(StarletteHTTPException)
async def friendly_http_error(request: Request, exc: StarletteHTTPException):
    """Owners never see JSON or a stack trace: errors get a short, plain page."""
    if _wants_json(request):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    heading, message = FRIENDLY.get(exc.status_code, ("Something went wrong", "Please try again in a minute."))
    if exc.status_code == 403 and request.url.path.startswith("/admin"):
        heading, message = "Admin link not valid", "Check the key in the address bar."
    return templates.TemplateResponse(request, "error.html", {"heading": heading, "message": message},
                                      status_code=exc.status_code)


@app.exception_handler(Exception)
async def friendly_crash(request: Request, exc: Exception):
    print(f"error on {request.url.path}: {type(exc).__name__}: {exc}", flush=True)  # details stay in the log
    if _wants_json(request):
        return JSONResponse({"error": "internal error"}, status_code=500)
    return templates.TemplateResponse(request, "error.html", {
        "heading": "Something went wrong", "message": "Please try again in a minute."}, status_code=500)


@app.get("/health")
def health():
    started = time.time()
    try:
        store.get_business("__health__")
    except Exception as exc:
        return JSONResponse({"ok": False, "store": store.BACKEND, "error": type(exc).__name__}, status_code=503)
    return {"ok": True, "store": store.BACKEND, "db_ms": round((time.time() - started) * 1000)}


def business_for(slug, token):
    """The business, or 404 for an unknown slug and 403 for a wrong token."""
    biz = store.get_business(slug)
    if biz is None:
        raise HTTPException(status_code=404, detail="Not found")
    expected = str(biz.get("preview_token") or "")
    if not expected or not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="This link is not valid")
    return biz


def payment_url(slug):
    """The Stripe payment link, tagged with the slug so we know who paid."""
    link = config.get("PAYMENT_LINK")
    if not link:
        return "#"
    separator = "&" if "?" in link else "?"
    return link + separator + urllib.parse.urlencode({"client_reference_id": slug})


def parse_date(iso):
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def tone(stars):
    """Pill color for a star rating: green for 4-5, amber for 3, red for 1-2."""
    if stars >= 4:
        return "green"
    return "amber" if stars == 3 else "red"


def ready_reviews(reviews):
    """Unanswered reviews that have a reply written, in the order given (newest first)."""
    return [r for r in reviews if r.get("owner_replied") is False and r.get("reply")]


def stats(reviews):
    unanswered = [r for r in reviews if r.get("owner_replied") is False]
    dates = [d for d in (parse_date(r.get("date")) for r in unanswered) if d]
    return {
        "unanswered": len(unanswered),
        "low_unanswered": sum(1 for r in unanswered if (r.get("stars") or 0) <= 2),
        "written": len(ready_reviews(reviews)),
        "oldest_days": (datetime.now(timezone.utc) - min(dates)).days if dates else None,
    }


def preview_order(ready):
    """Showcase first: one written review each of 1, 2, 3, 5 and 4 stars when available, topped up
    to five with other written reviews (lowest stars first). Then everything else, newest first."""
    written = [r for r in ready if (r.get("text") or "").strip()]
    showcase = []
    for stars in SHOWCASE_STARS:
        match = next((r for r in written if r.get("stars") == stars), None)
        if match is not None:
            showcase.append(match)
    for r in sorted(written, key=lambda r: r.get("stars") or 0):  # stable: newest first per star
        if len(showcase) >= FREE_CARDS:
            break
        if r not in showcase:
            showcase.append(r)
    return showcase + [r for r in ready if r not in showcase]


def card(review, locked=False):
    """Template data for one review card. A locked card gets lorem ipsum with the same word count
    instead of the reply, so the real text never reaches the browser."""
    stars = int(review.get("stars") or 0)
    author = (review.get("author") or "").strip() or "Google user"
    reply = review["reply"].strip()
    if locked:
        reply = " ".join(LOREM[i % len(LOREM)] for i in range(len(reply.split())))
    dt = parse_date(review.get("date"))
    return {
        "author": author,
        "initial": author[0].upper(),
        "stars": stars,
        "tone": tone(stars),
        "date": f"{dt:%b} {dt.day}, {dt.year}" if dt else "",
        "text": (review.get("text") or "").strip(),
        "reply": reply,
        "locked": locked,
    }


@app.get("/p/{slug}")
def preview(request: Request, slug: str, t: str = ""):
    biz = business_for(slug, t)
    if biz.get("paid"):  # a paying owner reopening the texted link gets the full set
        return RedirectResponse(f"/full/{urllib.parse.quote(slug)}?{urllib.parse.urlencode({'t': t})}",
                                status_code=303)
    reviews = store.reviews_for(slug)
    ordered = preview_order(ready_reviews(reviews))[:MAX_CARDS]
    cards = [card(r, locked=i >= FREE_CARDS) for i, r in enumerate(ordered)]
    return templates.TemplateResponse(request, "preview.html", {
        "biz": biz, "cards": cards, "pay_url": payment_url(slug), "price": config.PRICE_USD,
        **stats(reviews)})


@app.get("/p/{slug}/status.json")
def preview_status(slug: str, t: str = ""):
    """Polled by the preview page after payment; flips to paid once the business is marked paid."""
    return {"paid": bool(business_for(slug, t).get("paid"))}


@app.get("/full/{slug}")
def full(request: Request, slug: str, t: str = ""):
    biz = business_for(slug, t)
    if not biz.get("paid"):
        query = urllib.parse.urlencode({"t": t})
        return RedirectResponse(f"/p/{urllib.parse.quote(slug)}?{query}", status_code=303)
    reviews = store.reviews_for(slug)
    cards = [card(r) for r in ready_reviews(reviews)[:MAX_CARDS]]
    return templates.TemplateResponse(request, "full.html", {
        "biz": biz, "cards": cards, **stats(reviews)})


@app.get("/thanks")
def thanks(request: Request):
    return templates.TemplateResponse(request, "thanks.html", {})


# ---------------------------------------------------------------- admin

BUSINESS_ROWS = 300  # caps keep the dashboard fast however much we scrape
ORDER_ROWS = 50
STATS_TTL = 5  # seconds; every open admin tab polls stats.json
_pool = ThreadPoolExecutor(max_workers=4)
_stats_cache = {"at": 0.0, "value": None}


def cached_stats():
    if _stats_cache["value"] is None or time.time() - _stats_cache["at"] > STATS_TTL:
        _stats_cache.update(at=time.time(), value=store.stats())
    return _stats_cache["value"]


_evo = {"future": None}


def evolution_card(wait=0.3):
    """Prompt-evolution card data, or None while the run is going. The first load after
    evolve/results.json appears reads the store (and may make 2 LLM calls), so it runs in the
    background and the card says "in progress" until it is done; later loads come from memory."""
    if not evolution.RESULTS_PATH.exists():
        return None
    if evolution.cached():
        return evolution.load()
    future = _evo["future"]
    if future is None or future.done():
        future = _evo["future"] = _pool.submit(evolution.load)
    try:
        return future.result(timeout=wait)
    except FutureTimeout:
        return None


def require_admin(key):
    expected = config.get("ADMIN_KEY") or ""
    if not expected or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="This admin link is not valid")


def admin_numbers(s):
    """store.stats() plus what the dashboard derives from it."""
    tokens = (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
    cost = tokens / 1_000_000 * config.LLM_COST_PER_MILLION_TOKENS
    return dict(s,
                unanswered_pct=round(100 * s["unanswered"] / s["reviews"]) if s["reviews"] else 0,
                est_cost=round(cost, 4),
                cost_per_business=round(cost / s["businesses"], 5) if s["businesses"] else 0,
                reasoning_tokens=llm.logged_reasoning_tokens(),
                evolution_ready=evolution_card(wait=0.05) is not None)


def business_row(b):
    return {"slug": b["_id"], "name": b.get("name") or b["_id"], "category": b.get("category") or "",
            "phone": b.get("phone") or "", "unanswered": b.get("unanswered_count") or 0,
            "ready": b.get("replies_ready") or 0, "status": b.get("status") or "new",
            "paid": bool(b.get("paid")), "preview_url": outreach.preview_url(b), "text": outreach.outreach_text(b)}


def local_time(iso):
    dt = parse_date(iso)
    if not dt:
        return ""
    local = dt.astimezone()
    return f"{local:%b} {local.day}, {local.strftime('%I:%M %p').lstrip('0')}"


def back_to_admin(key):
    _stats_cache["value"] = None  # the next poll shows the change
    return RedirectResponse(f"/admin?{urllib.parse.urlencode({'key': key})}#businesses", status_code=303)


@app.get("/admin")
def admin(request: Request, key: str = ""):
    require_admin(key)
    stats_job = _pool.submit(store.stats)  # three Atlas round trips at once instead of in a row
    biz_job = _pool.submit(store.list_businesses)
    orders_job = _pool.submit(store.list_orders)
    businesses = biz_job.result()[:BUSINESS_ROWS]
    rows = [business_row(b) for b in businesses]
    names = {r["slug"]: r["name"] for r in rows}
    orders = [{"name": names.get(o["business_id"], o["business_id"]), "amount": o.get("amount") or 0,
               "method": o.get("method") or "", "when": local_time(o.get("paid_at"))}
              for o in orders_job.result()[:ORDER_ROWS]]
    first = [r for r in rows if r["status"] == "new" and r["phone"]][:5]  # rows come most unanswered first
    return templates.TemplateResponse(request, "admin.html", {
        "key": key, "stats": admin_numbers(stats_job.result()), "rows": rows, "first": first,
        "orders": orders, "price": config.PRICE_USD, "evo": evolution_card()})


@app.get("/admin/stats.json")
def admin_stats(key: str = ""):
    require_admin(key)
    return admin_numbers(cached_stats())


@app.post("/admin/mark-paid/{slug}")
def admin_mark_paid(slug: str, key: str = ""):
    require_admin(key)
    if store.get_business(slug) is None:
        raise HTTPException(status_code=404, detail="Not found")
    store.mark_paid(slug, "stripe", config.PRICE_USD)
    return back_to_admin(key)


@app.post("/admin/mark-texted/{slug}")
def admin_mark_texted(slug: str, key: str = ""):
    require_admin(key)
    if store.get_business(slug) is None:
        raise HTTPException(status_code=404, detail="Not found")
    store.mark_texted(slug)
    return back_to_admin(key)


@app.get("/admin/export.csv")
def admin_export(key: str = ""):
    require_admin(key)
    return Response("﻿" + outreach.csv_text(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="replydesk-outreach.csv"'})
