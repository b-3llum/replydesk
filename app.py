"""ReplyDesk web app: the owner's preview page, the paid full set, and the Stripe return page.

    python run.py serve          (uvicorn on 0.0.0.0:8000)

    GET /p/{slug}?t=token        preview: 5 replies readable, the rest locked (real text never sent)
    GET /full/{slug}?t=token     every reply, once the business has paid
    GET /thanks                  where Stripe sends the owner after paying
"""
import secrets
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import store

SHOWCASE_STARS = (1, 2, 3, 5, 4)  # the preview opens with one written review of each, in this order
FREE_CARDS = 5  # readable cards on the preview; every card after these is locked
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
    ordered = preview_order(ready_reviews(reviews))
    cards = [card(r, locked=i >= FREE_CARDS) for i, r in enumerate(ordered)]
    return templates.TemplateResponse(request, "preview.html", {
        "biz": biz, "cards": cards, "pay_url": payment_url(slug), "price": config.PRICE_USD,
        **stats(reviews)})


@app.get("/full/{slug}")
def full(request: Request, slug: str, t: str = ""):
    biz = business_for(slug, t)
    if not biz.get("paid"):
        query = urllib.parse.urlencode({"t": t})
        return RedirectResponse(f"/p/{urllib.parse.quote(slug)}?{query}", status_code=303)
    reviews = store.reviews_for(slug)
    cards = [card(r) for r in ready_reviews(reviews)]
    return templates.TemplateResponse(request, "full.html", {
        "biz": biz, "cards": cards, **stats(reviews)})


@app.get("/thanks")
def thanks(request: Request):
    return templates.TemplateResponse(request, "thanks.html", {})
