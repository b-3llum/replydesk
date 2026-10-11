"""Snapshots everything the evolve results depend on into evolve/exports/ (local only: it holds
review text and reviewer names, and the repo is public), plus evolve/summary.json, the
numbers-only results that are committed.

    python -m evolve.export_data

Writes to exports/:
    reviews.json          every review used (train / holdout / fresh validation), with its set
    businesses.json       the businesses those reviews belong to
    prompt_runs.jsonl     the Mongo prompt_runs collection: every scored prompt, per-review replies
    blind_judgments.json  the independent judge's votes, un-blinded (which prompt was A / B)
"""
import json

from evolve.common import EVOLVE_DIR, db
from evolve.score import POINTS, load_split

OUT = EVOLVE_DIR / "exports"


def _json_default(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def main():
    OUT.mkdir(exist_ok=True)
    split = load_split()
    validation = json.loads((EVOLVE_DIR / "validation.json").read_text(encoding="utf-8"))
    sets = {"train": split["train"], "holdout": split["holdout"], "fresh": validation["fresh_review_ids"]}
    review_set = {rid: name for name, ids in sets.items() for rid in ids}

    fields = {"business_id": 1, "author": 1, "stars": 1, "date": 1, "text": 1}
    reviews = sorted(db().reviews.find({"_id": {"$in": list(review_set)}}, fields),
                     key=lambda r: (list(sets).index(review_set[r["_id"]]), r["_id"]))
    for r in reviews:
        r["set"] = review_set[r["_id"]]
    missing = set(review_set) - {r["_id"] for r in reviews}
    if missing:
        raise SystemExit(f"{len(missing)} reviews are no longer in Mongo, e.g. {sorted(missing)[:2]}")
    biz_ids = sorted({r["business_id"] for r in reviews})
    businesses = list(db().businesses.find({"_id": {"$in": biz_ids}},
                                           {"name": 1, "category": 1, "phone": 1, "address": 1}))

    (OUT / "reviews.json").write_text(json.dumps(reviews, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "businesses.json").write_text(json.dumps(businesses, indent=1, ensure_ascii=False), encoding="utf-8")

    runs = list(db().prompt_runs.find().sort([("created_at", 1)]))
    with open(OUT / "prompt_runs.jsonl", "w", encoding="utf-8") as f:
        for doc in runs:
            f.write(json.dumps(doc, default=_json_default, ensure_ascii=False) + "\n")

    pairs = {p["pair_id"]: p for p in json.loads((EVOLVE_DIR / "blind_pairs.json").read_text(encoding="utf-8"))}
    judgments = []
    for v in validation["blind_judge"]["votes"]:
        p = pairs[v["pair_id"]]
        judgments.append({"pair_id": v["pair_id"], "comparison": f"best vs {p['comparison']}",
                          "review_id": p["review_id"], "stars": p["stars"], "A_is": p["A_is"], "B_is": p["B_is"],
                          "winner": v["winner"], "winner_prompt": p[f"{v['winner']}_is"] if v["winner"] != "tie" else "tie",
                          "reason": v["reason"]})
    (OUT / "blind_judgments.json").write_text(json.dumps(judgments, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"exports: {len(reviews)} reviews ({', '.join(f'{k} {len(v)}' for k, v in sets.items())}), "
          f"{len(businesses)} businesses, {len(runs)} prompt_runs docs, {len(judgments)} blind judgments -> {OUT}")
    print(f"wrote {write_summary()}")


def write_summary():
    """evolve/summary.json: numbers only (no review text, names, replies or prompt text), safe to publish."""
    results = json.loads((EVOLVE_DIR / "results.json").read_text(encoding="utf-8"))
    validation = json.loads((EVOLVE_DIR / "validation.json").read_text(encoding="utf-8"))

    def rates(detail):
        rows = list(detail.values())
        out = {k: round(sum(1 for r in rows if k not in r["failed"]) / len(rows), 3) for k in POINTS}
        out["judge_mean"] = round(sum(r["judge"] for r in rows) / len(rows), 2)
        return out

    rounds = [{"round": r["round"], "best_train": r["best_train"], "best_id": r["best_id"],
               "mean_new": r["mean_new"], "best_strategy": r["best_strategy"],
               "candidates": [{k: c.get(k) for k in ("strategy", "id", "parent_ids", "score", "duplicate")}
                              for c in r["candidates"]]}
              for r in results["rounds"]]
    blind = validation.get("blind_judge", {})
    summary = {
        "run_id": results["run_id"], "model": results["model"], "judge_version": results["judge_version"],
        "split": {"train_reviews": 40, "holdout_reviews": 30, "fresh_reviews": len(validation["fresh_review_ids"]),
                  "fresh_businesses": len(validation["fresh_businesses"])},
        "run": {"baseline_train": results["baseline_train"], "baseline_holdout": results["baseline_holdout"],
                "best_train": results["best_train"], "best_holdout": results["best_holdout"],
                "best_prompt_id": results["best_prompt_id"], "best_strategy": results["best_strategy"],
                "best_round": results["best_round"], "rounds": rounds,
                "holdout_pass_rates": {name: rates(d) for name, d in results["holdout_detail"].items()}},
        "fresh_validation": {"round1_prompt_id": validation["round1_prompt_id"], "scores": validation["scores"],
                             "best_vs": validation["best_vs"]},
        "blind_judge": {"judge": blind.get("judge"),
                        "tally": {name: {k: v for k, v in t.items()} for name, t in blind.get("tally", {}).items()}},
    }
    path = EVOLVE_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
