"""ReplyDesk smoke checks: a plain script, no test framework.

    python smoke.py            run every section
    python smoke.py store      run only the named section(s)

Prints one line per check (PASS <name> / FAIL <name>: <detail> / SKIP <name>: <reason>),
then a SUMMARY line. Exit code 1 if anything failed.

To add a section: write a section_<name>() function and register it in SECTIONS at the
bottom. Import project modules other than store inside the section function, so one broken
module can't stop the other sections from running.
"""
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import config
import store

_counts = {"pass": 0, "fail": 0, "skip": 0}


def check(name, ok, detail=""):
    """Print and count one check. Returns ok."""
    if ok:
        _counts["pass"] += 1
        print(f"PASS {name}", flush=True)
    else:
        _counts["fail"] += 1
        detail = store.redact(detail)
        if len(detail) > 600:
            detail = detail[:600] + "..."
        print(f"FAIL {name}: {detail}", flush=True)
    return bool(ok)


def skip(name, reason):
    _counts["skip"] += 1
    print(f"SKIP {name}: {store.redact(reason)}", flush=True)


def run_guarded(name, fn, *args):
    """Run a group of checks; a crash becomes one FAIL line instead of ending the script."""
    try:
        fn(*args)
    except Exception as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        check(f"{name} (crashed)", False,
              f"{type(e).__name__}: {e} at {os.path.basename(frame.filename)}:{frame.lineno}")


def _by_id(docs):
    return {d["_id"]: d for d in docs}


def _file_sig(path):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _utc_iso(value, recent=False):
    """True if value is an ISO-8601 UTC string (and with recent=True, within 10 minutes of now)."""
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    if dt.utcoffset() != timedelta(0):
        return False
    return not recent or abs(datetime.now(timezone.utc) - dt) < timedelta(minutes=10)


# ---------------------------------------------------------------- section: store

def _review(rid, slug, date, text, owner_replied=False, **extra):
    doc = {"_id": rid, "business_id": slug, "author": "Pat Q.", "stars": 5, "text": text,
           "date": date, "owner_replied": owner_replied}
    doc.update(extra)
    return doc


def check_api(s, label):
    """Behavior checks shared by every backend. s is a JsonStore, a MongoStore or the store module."""
    lifecycle = ("preview_token", "paid", "status", "created_at")
    joes = {"_id": "joes-pizza", "name": "Joe's Pizza", "category": "Pizza restaurant",
            "address": "1 Old York Rd, Abington, PA 19001", "phone": "(215) 555-0101",
            "maps_url": "https://maps.google.com/?cid=1", "rating": 4.1, "review_count": 87,
            "unanswered_count": 2, "place_id": "ChIJ-joes"}
    biz = s.upsert_business(dict(joes))
    check(f"{label}: upsert_business inserts defaults",
          isinstance(biz.get("preview_token"), str) and len(biz["preview_token"]) >= 8
          and biz.get("paid") is False and biz.get("status") == "new"
          and _utc_iso(biz.get("created_at"), recent=True)
          and biz.get("name") == "Joe's Pizza" and biz.get("place_id") == "ChIJ-joes", biz)

    # Bella already has history (created long ago, texted), so a reset would be visible.
    s.upsert_business({"_id": "bella-salon", "name": "Bella Salon", "rating": 4.5,
                       "unanswered_count": 5, "created_at": "2020-01-01T00:00:00+00:00"})
    s.mark_texted("bella-salon")
    before = s.get_business("bella-salon")
    after = s.upsert_business({"_id": "bella-salon", "rating": 4.8})
    check(f"{label}: re-upsert with a new rating keeps preview_token/paid/status/created_at",
          after.get("rating") == 4.8 and after.get("name") == "Bella Salon"
          and all(after.get(k) == before.get(k) for k in lifecycle)
          and after.get("status") == "texted" and str(after.get("created_at")).startswith("2020"),
          f"before={before} after={after}")
    after = s.upsert_business({"_id": "bella-salon", "rating": 4.9, "paid": False, "status": "new",
                               "preview_token": "regenerated", "created_at": "2099-01-01T00:00:00+00:00"})
    check(f"{label}: a re-scrape that sends lifecycle fields still can't reset them",
          after.get("rating") == 4.9 and all(after.get(k) == before.get(k) for k in lifecycle), after)

    s.upsert_business({"_id": "abc-auto", "name": "ABC Auto Repair", "unanswered_count": 5})
    order = [b["_id"] for b in s.list_businesses()]
    check(f"{label}: list_businesses sorted by unanswered_count desc, then name",
          order == ["abc-auto", "bella-salon", "joes-pizza"], order)

    reviews = [
        _review("joes-r1", "joes-pizza", "2026-10-01T18:00:00+00:00",
                "Best slice in Abington \U0001F355 très bon", likes=4),
        _review("joes-r2", "joes-pizza", "2026-09-15T12:00:00+00:00", "Delivery was slow"),
        _review("joes-r3", "joes-pizza", None, "Good"),
        _review("joes-r4", "joes-pizza", "2026-10-05T09:30:00+00:00", "Owner was lovely",
                owner_replied=True),
        _review("bella-r1", "bella-salon", "2026-08-01T10:00:00+00:00", "Great cut, señora!"),
    ]
    count, empty = s.upsert_reviews(reviews), s.upsert_reviews([])
    r1 = _by_id(s.reviews_for("joes-pizza")).get("joes-r1", {})
    check(f"{label}: upsert_reviews inserts empty reply fields and keeps extra keys",
          count == 5 and empty == 0
          and all(k in r1 and r1[k] == v for k, v in store.REVIEW_DEFAULTS.items())
          and r1.get("likes") == 4 and r1.get("text") == reviews[0]["text"],
          f"count={count} empty={empty} r1={r1}")

    slug_ids = [r["_id"] for r in s.pending_reviews("joes-pizza")]
    all_ids = [r["_id"] for r in s.pending_reviews()]
    check(f"{label}: pending_reviews with and without slug",
          slug_ids == ["joes-r1", "joes-r2", "joes-r3"]
          and all_ids == ["joes-r1", "joes-r2", "bella-r1", "joes-r3"],
          f"slug={slug_ids} all={all_ids}")

    s.save_reply("joes-r1", "Thank you so much! \U0001F355", "llama-3.1-8b", 120, 45)
    s.save_reply("joes-r2", "Thanks for the feedback.", "template", 0, 0)
    # Second scrape: edited text, owner replied to r2; stray reply keys must be ignored.
    count = s.upsert_reviews([
        dict(reviews[0], text="Best slice in Abington (edited)", reply=None, model=None, tokens_in=0),
        dict(reviews[1], owner_replied=True),
    ])
    got = _by_id(s.reviews_for("joes-pizza"))
    r1, r2 = got.get("joes-r1", {}), got.get("joes-r2", {})
    check(f"{label}: re-upsert_reviews updates text/owner_replied, keeps reply/model/tokens",
          count == 2 and r1.get("text") == "Best slice in Abington (edited)" and r1.get("likes") == 4
          and r1.get("reply") == "Thank you so much! \U0001F355" and r1.get("model") == "llama-3.1-8b"
          and r1.get("tokens_in") == 120 and r1.get("tokens_out") == 45
          and _utc_iso(r1.get("generated_at"), recent=True)
          and r2.get("owner_replied") is True and r2.get("reply") == "Thanks for the feedback."
          and r2.get("model") == "template",
          f"count={count} r1={r1} r2={r2}")

    slug_ids = [r["_id"] for r in s.pending_reviews("joes-pizza")]
    all_ids = [r["_id"] for r in s.pending_reviews()]
    check(f"{label}: pending_reviews drops replied and owner-answered reviews",
          slug_ids == ["joes-r3"] and all_ids == ["bella-r1", "joes-r3"],
          f"slug={slug_ids} all={all_ids}")

    expected = {"businesses": 3, "reviews": 5, "unanswered": 3, "replies_ready": 1,
                "paid_orders": 0, "revenue": 0, "orders_today": 0, "revenue_today": 0,
                "tokens_in": 120, "tokens_out": 45, "llm_calls": 1}
    got = s.stats()
    check(f"{label}: stats() exact after save_reply (llm_calls excludes template)",
          got == expected, f"got {got}")

    biz = s.get_business("joes-pizza")
    listed = _by_id(s.list_businesses())
    check(f"{label}: get_business and list_businesses include replies_ready",
          biz.get("replies_ready") == 1 and listed["joes-pizza"].get("replies_ready") == 1
          and listed["bella-salon"].get("replies_ready") == 0
          and listed["abc-auto"].get("replies_ready") == 0 and s.get_business("no-such-biz") is None,
          f"get={biz} list={[(k, v.get('replies_ready')) for k, v in listed.items()]}")

    first, second = s.mark_paid("joes-pizza"), s.mark_paid("joes-pizza")
    orders = s.list_orders()
    order = orders[0] if orders else {}
    biz = s.get_business("joes-pizza")
    check(f"{label}: mark_paid returns True then False and creates exactly one order",
          first is True and second is False and len(orders) == 1
          and order.get("business_id") == "joes-pizza" and order.get("amount") == 29
          and order.get("method") == "stripe" and _utc_iso(order.get("paid_at"), recent=True)
          and isinstance(order.get("_id"), str) and biz.get("paid") is True
          and biz.get("status") == "paid",
          f"first={first} second={second} orders={orders} biz={biz}")
    got = s.stats()
    check(f"{label}: stats() counts the order as today's revenue",
          got == dict(expected, paid_orders=1, revenue=29, orders_today=1, revenue_today=29),
          f"got {got}")

    texted = s.mark_texted("joes-pizza")
    check(f"{label}: mark_texted does not downgrade a paid business",
          texted is False and s.get_business("joes-pizza").get("status") == "paid", texted)
    check(f"{label}: mark_texted sets texted; unknown slugs return False",
          s.mark_texted("abc-auto") is True and s.get_business("abc-auto").get("status") == "texted"
          and s.mark_texted("no-such-biz") is False and s.mark_paid("no-such-biz") is False)

    s.upsert_business(dict(joes, rating=4.3))
    biz = s.get_business("joes-pizza")
    check(f"{label}: a re-scrape after payment keeps paid",
          biz.get("paid") is True and biz.get("status") == "paid" and biz.get("rating") == 4.3, biz)

    order = [r["_id"] for r in s.reviews_for("joes-pizza")]
    check(f"{label}: reviews_for newest first, None date last",
          order == ["joes-r4", "joes-r1", "joes-r2", "joes-r3"], order)

    doc = {"_id": "copy-cafe", "name": "Copy Cafe", "tags": ["coffee"]}
    s.upsert_business(doc)
    doc["tags"].append("mutated")
    s.get_business("copy-cafe")["name"] = "mutated"
    s.reviews_for("joes-pizza")[0]["text"] = "mutated"
    s.list_businesses()[0]["name"] = "mutated"
    s.list_orders()[0]["amount"] = 0
    fresh = s.get_business("copy-cafe")
    check(f"{label}: docs passed in and returned are copies",
          fresh.get("name") == "Copy Cafe" and fresh.get("tags") == ["coffee"]
          and s.reviews_for("joes-pizza")[0]["text"] != "mutated"
          and all(b.get("name") != "mutated" for b in s.list_businesses())
          and s.list_orders()[0]["amount"] == 29, fresh)

    try:
        s.save_reply("no-such-review", "x", "m", 1, 1)
        raised = "nothing"
    except KeyError:
        raised = "KeyError"
    except Exception as e:
        raised = type(e).__name__
    check(f"{label}: save_reply on an unknown review raises KeyError", raised == "KeyError", raised)


def check_import_is_lazy():
    code = "import store; print(store._backend is None, store.BACKEND)"
    t0 = time.monotonic()
    out = subprocess.run([sys.executable, "-c", code], cwd=str(config.BASE_DIR),
                         capture_output=True, text=True, timeout=60)
    check(f"store: import creates no backend ({time.monotonic() - t0:.1f}s, BACKEND={store.BACKEND})",
          out.returncode == 0 and out.stdout.split() == ["True", store.BACKEND],
          (out.stderr.strip() or out.stdout.strip())[-400:])


def check_module_api(tmp):
    """The module-level functions, pointed at a temp JSON store instead of the real backend."""
    saved = store._backend
    store._backend = store.JsonStore(os.path.join(tmp, "module.json"))
    try:
        check_api(store, "module")
    finally:
        store._backend = saved


def check_json_threads(tmp):
    path = os.path.join(tmp, "threads.json")
    s = store.JsonStore(path)
    s.upsert_business({"_id": "thread-biz", "name": "Thread Biz"})
    s.upsert_reviews([_review(f"t{i}", "thread-biz", f"2026-01-01T00:{i // 60 % 60:02d}:{i % 60:02d}+00:00",
                              "fine \U0001F44D") for i in range(1000)])
    errors, writing = [], threading.Event()

    def work(start):
        try:
            for i in range(start, start + 25):
                s.save_reply(f"t{i}", f"reply {i} ✓", "model-thread", 7, 3)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")

    def insert_more():  # grows the reviews dict while readers copy it
        try:
            for k in range(20):
                s.upsert_reviews([_review(f"x{k}-{j}", "thread-biz-2", None, "new") for j in range(10)])
        except Exception as e:
            errors.append(f"insert {type(e).__name__}: {e}")

    def read_loop():
        try:
            while writing.is_set():
                s.pending_reviews()
                s.stats()
                time.sleep(0.005)
        except Exception as e:
            errors.append(f"read {type(e).__name__}: {e}")

    threads = [threading.Thread(target=work, args=(k * 25,)) for k in range(40)]
    threads.append(threading.Thread(target=insert_more))
    readers = [threading.Thread(target=read_loop) for _ in range(2)]
    writing.set()
    t0 = time.monotonic()
    for t in threads + readers:
        t.start()
    for t in threads:
        t.join()
    writing.clear()
    for t in readers:
        t.join()
    check(f"json threads: 40 threads x 25 save_reply + inserts + readers, no errors "
          f"({time.monotonic() - t0:.1f}s)", not errors, "; ".join(errors[:3]))
    fresh_store = store.JsonStore(path)
    fresh = _by_id(fresh_store.reviews_for("thread-biz"))
    ok = sum(1 for i in range(1000) if fresh.get(f"t{i}", {}).get("reply") == f"reply {i} ✓")
    extra = len(fresh_store.reviews_for("thread-biz-2"))
    check("json threads: a fresh JsonStore sees all 1000 replies and 200 inserted reviews",
          ok == 1000 and extra == 200, f"{ok}/1000 replies, {extra}/200 inserted")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        detail = sum(1 for r in raw["reviews"].values() if r.get("reply"))
    except (OSError, ValueError, KeyError) as e:
        detail = f"{type(e).__name__}: {e}"
    check("json threads: the store file parses and holds 1000 replies", detail == 1000, detail)


def _save_paced(s, start, count):
    """save_reply with short random gaps (like LLM latency), so the writer processes interleave
    instead of each holding the lock for its whole run."""
    for i in range(start, start + count):
        s.save_reply(f"p{i}", f"reply {i}", "model-proc", 3, 4)
        time.sleep(random.uniform(0, 0.01))


def _writer(path, start, count):
    """Child process for the multi-process check: wait for 'go', then save count replies."""
    s = store.JsonStore(path)
    print("ready", flush=True)
    sys.stdin.readline()
    _save_paced(s, int(start), int(count))
    return 0


def check_json_processes(tmp):
    path = os.path.join(tmp, "procs.json")
    s = store.JsonStore(path)
    s.upsert_business({"_id": "proc-biz", "name": "Proc Biz"})
    s.upsert_reviews([_review(f"p{i}", "proc-biz", None, "fine") for i in range(450)])
    cmd = [sys.executable, os.path.abspath(__file__), "_writer", path]
    procs = [subprocess.Popen(cmd + [str(start), "150"], cwd=str(config.BASE_DIR),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace")
             for start in (150, 300)]
    for p in procs:
        p.stdout.readline()  # "ready": the child has loaded the store

    # Like the web server: another store instance reading while the others write.
    stop, reads, read_errors = threading.Event(), [0], []

    def reader():
        r = store.JsonStore(path)
        while not stop.is_set():
            try:
                r.stats()
                reads[0] += 1
            except Exception as e:
                read_errors.append(f"{type(e).__name__}: {e}")
                return

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    t0 = time.monotonic()
    problems = []
    try:
        for p in procs:  # start everyone at once so the writers really overlap
            try:
                p.stdin.write("go\n")
                p.stdin.flush()
            except OSError:
                pass  # child already died; its exit code is reported below
        _save_paced(s, 0, 150)
    finally:
        for p in procs:
            try:
                _, err = p.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
                _, err = p.communicate()
            if p.returncode != 0:
                last = err.strip().splitlines()[-1] if err.strip() else ""
                problems.append(f"writer exit {p.returncode}: {last}")
        stop.set()
        reader_thread.join(timeout=30)
    check(f"json processes: 2 writer subprocesses + parent finish cleanly ({time.monotonic() - t0:.1f}s)",
          not problems, "; ".join(problems))
    check(f"json processes: a concurrent reader never failed ({reads[0]} reads)",
          not read_errors and reads[0] > 0, "; ".join(read_errors[:3]) or "no reads")
    fresh = _by_id(store.JsonStore(path).reviews_for("proc-biz"))
    ok = sum(1 for i in range(450) if fresh.get(f"p{i}", {}).get("reply") == f"reply {i}")
    check("json processes: all 450 replies present in a fresh JsonStore", ok == 450, f"{ok}/450")
    seen = sum(1 for r in s.reviews_for("proc-biz") if r.get("reply"))
    check("json processes: the parent's own store sees the children's writes", seen == 450, f"{seen}/450")


def check_json_lock_and_recovery(tmp):
    path = os.path.join(tmp, "lock.json")
    s = store.JsonStore(path)
    s.upsert_business({"_id": "lock-biz", "name": "Lock Biz"})

    with open(s.lock_path, "w") as f:  # another process holds the lock...
        f.write("12345")
    threading.Timer(0.3, os.remove, [s.lock_path]).start()  # ...and releases it 0.3s later
    t0 = time.monotonic()
    s.upsert_business({"_id": "lock-biz", "rating": 4.0})
    waited = time.monotonic() - t0
    check(f"json lock: a held lock file blocks writers until released (waited {waited:.2f}s)",
          waited >= 0.25 and s.get_business("lock-biz").get("rating") == 4.0, f"waited {waited:.2f}s")

    with open(s.lock_path, "w") as f:  # left behind by a crashed process
        f.write("99999")
    old = time.time() - 60
    os.utime(s.lock_path, (old, old))
    t0 = time.monotonic()
    s.upsert_business({"_id": "lock-biz", "rating": 4.2})
    waited = time.monotonic() - t0
    check(f"json lock: a stale lock file (>30s old) is broken ({waited:.2f}s)",
          waited < 2 and s.get_business("lock-biz").get("rating") == 4.2
          and not os.path.exists(s.lock_path), f"waited {waited:.2f}s")

    garbage = '{"businesses": {"broken'
    with open(path, "w", encoding="utf-8") as f:  # e.g. a bad hand edit
        f.write(garbage)
    biz = s.get_business("lock-biz")
    try:
        s.upsert_business({"_id": "lock-biz", "rating": 1.0})
        refused = "no error"
    except RuntimeError:
        refused = "RuntimeError"
    with open(path, encoding="utf-8") as f:
        untouched = f.read() == garbage
    check("json: a corrupt file keeps serving the last good copy and is never overwritten",
          (biz or {}).get("rating") == 4.2 and refused == "RuntimeError" and untouched,
          f"biz={biz} write={refused} file_untouched={untouched}")


def check_mongo(tmp):
    uri = config.get("MONGODB_URI")
    if not uri:
        skip("mongo", "MONGODB_URI is not set")
        return
    db_api = "replydesk_smoke_" + secrets.token_hex(3)
    db_migrate = "replydesk_smoke_" + secrets.token_hex(3)
    m = None
    try:
        try:
            m = store.MongoStore(uri, db_api)
        except RuntimeError as e:
            skip("mongo", str(e))
            return
        check_api(m, "mongo")

        src_path = os.path.join(tmp, "migrate.json")
        src = store.JsonStore(src_path)
        src.upsert_business({"_id": "mig-cafe", "name": "Migration Cafe", "unanswered_count": 2})
        src.upsert_business({"_id": "mig-deli", "name": "Migration Deli", "unanswered_count": 1})
        src.upsert_reviews([
            _review("mig-r1", "mig-cafe", "2026-10-01T08:00:00+00:00", "Lovely latte ☕"),
            _review("mig-r2", "mig-cafe", None, "ok"),
            _review("mig-r3", "mig-deli", "2026-09-01T12:00:00+00:00", "Fine"),
        ])
        src.save_reply("mig-r1", "Thanks for stopping by ☕", "llama-3.1-8b", 50, 20)
        src.mark_paid("mig-cafe")
        token = src.get_business("mig-cafe")["preview_token"]

        counts = store.migrate_json_to_mongo(src_path, uri, db_migrate)
        check("mongo: migrate_json_to_mongo counts",
              counts == {"businesses": 2, "reviews": 3, "orders_inserted": 1, "orders_skipped": 0}, counts)
        counts = store.migrate_json_to_mongo(src_path, uri, db_migrate)
        check("mongo: migrate again is idempotent (no duplicate order)",
              counts == {"businesses": 2, "reviews": 3, "orders_inserted": 0, "orders_skipped": 1}, counts)
        dst = store.MongoStore(uri, db_migrate)
        try:
            r1 = _by_id(dst.reviews_for("mig-cafe")).get("mig-r1", {})
            biz = dst.get_business("mig-cafe") or {}
            got = dst.stats()
            check("mongo: migrated reply, preview_token and payment survive",
                  r1.get("reply") == "Thanks for stopping by ☕" and r1.get("model") == "llama-3.1-8b"
                  and r1.get("tokens_in") == 50 and biz.get("preview_token") == token
                  and biz.get("paid") is True and biz.get("replies_ready") == 1
                  and got["businesses"] == 2 and got["reviews"] == 3 and got["paid_orders"] == 1
                  and len(dst.list_orders()) == 1,
                  f"r1={r1} biz={biz} stats={got}")
        finally:
            dst.close()
    finally:
        if m is not None:
            for name in (db_api, db_migrate):
                try:
                    m.client.drop_database(name)
                except Exception as e:
                    check(f"mongo: drop throwaway db {name}", False, f"{type(e).__name__}: {e}")
            m.close()


def section_store():
    """store.py on both backends, plus thread, process and migration safety."""
    real_before = _file_sig(store.JSON_PATH)
    tmp = tempfile.mkdtemp(prefix="replydesk_smoke_")
    try:
        run_guarded("store: import", check_import_is_lazy)
        run_guarded("json", lambda: check_api(store.JsonStore(os.path.join(tmp, "api.json")), "json"))
        run_guarded("module", check_module_api, tmp)
        run_guarded("json threads", check_json_threads, tmp)
        run_guarded("json processes", check_json_processes, tmp)
        run_guarded("json lock", check_json_lock_and_recovery, tmp)
        leftovers = [f for f in os.listdir(tmp) if f.endswith((".lock", ".tmp"))]
        check("json: no leftover lock or temp files", not leftovers, leftovers)
        run_guarded("mongo", check_mongo, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("store: the real data/store.json was not touched", _file_sig(store.JSON_PATH) == real_before)


def section_llm():
    """prompts.py and generate.py rules offline, then one live Luminal call with thinking off."""
    import generate
    import llm
    import prompts

    biz = {"name": "Test Diner", "category": "Diner", "phone": "(215) 555-0100"}
    msgs = prompts.build_messages(biz, {"author": "Sam Lee", "stars": 1,
                                        "text": "Cold soup and a 40 minute wait for the check."})
    check("prompts: system + 3 few-shot pairs + the review", len(msgs) == 8
          and [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"])
    for (_, reply), stars in zip(prompts.FEW_SHOTS, (5, 3, 1)):
        broken = generate.problems(reply, {"stars": stars}, "(215) 555-0142")
        check(f"prompts: the {stars}-star few-shot follows the rules", not broken, broken)
    for stars in (5, 3, 1):
        for phone in ("(215) 555-0100", ""):
            review = {"_id": f"t{stars}", "author": "kevin T.", "stars": stars, "text": ""}
            for variant in range(4):
                review["_id"] = f"t{stars}{variant}"
                text = generate.fill_phone(generate.template_reply(review, variant), phone)
                check(f"generate: {stars}-star template {variant}, phone={bool(phone)}: no placeholder, capitalized",
                      "[" not in text and "{" not in text and text[0].isupper() and "Kevin" in text, text)
    highs = [generate.template_reply({"_id": "x", "author": "", "stars": 5, "text": ""}, v) for v in range(4)]
    check("generate: 4 distinct 5-star templates, none says 'whole team'",
          len(set(highs)) == 4 and not any("whole team" in h for h in highs), highs)
    low2 = generate.fill_phone(generate.template_reply({"_id": "x", "author": "", "stars": 1, "text": ""}, 1), "")
    check("generate: no-phone low template reads 'stop by'", "stop by" in low2 and " at us" not in low2, low2)
    check("generate: opening_key", generate.opening_key("Jade, you ordered three veg tacos.")
          == "jade you ordered three")
    check("generate: closing_key", generate.closing_key("Sorry. I'll fix it. Hope to see you soon!")
          == "hope to see you soon")
    call = generate.call_key("I'm sorry. I'd like to fix it, so please call us at (215) 796-9890 and ask for the owner.")
    check("generate: call_key", call == "call us at zzphone and ask for the owner", call)

    out = llm.complete(msgs, tag="smoke")
    check("llm: live reply is non-empty", bool(out["text"]), out)
    check("llm: thinking is off (reasoning_tokens == 0)", out["reasoning_tokens"] == 0, out)
    check("llm: token usage counted", out["tokens_in"] > 0 and out["tokens_out"] > 0, out)


# ---------------------------------------------------------------- main

SECTIONS = {
    "store": section_store,
    "llm": section_llm,
}


def main(argv):
    if argv[:1] == ["_writer"]:
        return _writer(*argv[1:4])
    names = argv or list(SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}. Sections: {', '.join(SECTIONS)}")
        return 2
    for name in names:
        run_guarded(name, SECTIONS[name])
    print(f"SUMMARY {_counts['pass']} passed, {_counts['fail']} failed, {_counts['skip']} skipped",
          flush=True)
    return 1 if _counts["fail"] else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main(sys.argv[1:]))
