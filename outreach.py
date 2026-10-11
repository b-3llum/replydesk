"""Outreach: who to text first, their preview links, and the CSV export."""
import csv
import io

import config
import store

COLUMNS = ["name", "phone", "category", "unanswered", "preview_url", "status"]
TEXT = ("Hi, this is Moussa, Penn State student from Philly. Your business has {n} Google reviews with no "
        "reply, which hurts your ranking. We already wrote them for you: {url}")


def preview_url(biz):
    return f"{config.require('PUBLIC_BASE_URL').rstrip('/')}/p/{biz['_id']}?t={biz['preview_token']}"


def outreach_text(biz):
    return TEXT.format(n=biz.get("unanswered_count") or 0, url=preview_url(biz))


def rows(businesses=None):
    """One row per business, most unanswered reviews first."""
    businesses = store.list_businesses() if businesses is None else businesses
    out = [{"name": b.get("name") or "", "phone": b.get("phone") or "", "category": b.get("category") or "",
            "unanswered": b.get("unanswered_count") or 0, "preview_url": preview_url(b),
            "status": b.get("status") or "new"} for b in businesses]
    return sorted(out, key=lambda r: (-r["unanswered"], r["name"]))


def csv_text(businesses=None):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows(businesses))
    return buf.getvalue()


def write_csv(path=None):
    """Write data/outreach.csv (with a BOM so Excel shows accented names right). Returns the path."""
    path = path or config.DATA_DIR / "outreach.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(), encoding="utf-8-sig", newline="")
    return path
