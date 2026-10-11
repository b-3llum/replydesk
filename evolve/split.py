"""Builds evolve/split.json: 40 training and 30 held-out reviews from disjoint businesses.

    python -m evolve.split           # refuses to overwrite an existing split
    python -m evolve.split --force

Rating-only reviews (no text) are excluded. Businesses are fixed below so the hold-out
covers different restaurants; each side keeps enough 1-2 star reviews. Sampling is
stratified by stars with a fixed seed, so the same Mongo data gives the same ids.
"""
import argparse
import json
import random
import sys
from collections import Counter

from evolve.common import SPLIT_PATH, db, has_text

SEED = 20261011
# Text-review 1-2 star counts at split time: loco-pez 17, dahlak 6, bookers 4, mood 2,
# devs 0 (train = 29) / clarkville 9, alif 5, doro 4, vietnam 2, bayou 1 (holdout = 21).
TRAIN_BUSINESSES = ["loco-pez-west-philly-b71a", "dahlak-83dd", "bookers-restaurant-bar-a6b4",
                    "mood-cafe-f390", "devs-best-indian-restaurant-cdba"]
HOLDOUT_BUSINESSES = ["clarkville-d9a6", "alif-brew-mini-mart-inc-16c8", "doro-bet-004c",
                      "vietnam-cafe-45ff", "bayou-lounge-and-bar-cb0e"]
# reviews per star bucket: (1-2 stars, 3, 4, 5)
TRAIN_QUOTA = {"low": 14, 3: 6, 4: 6, 5: 14}      # 40
HOLDOUT_QUOTA = {"low": 10, 3: 5, 4: 5, 5: 10}    # 30


def _bucket(stars):
    return "low" if stars <= 2 else stars


def _sample(businesses, quota, rng):
    reviews = [r for r in db().reviews.find({"business_id": {"$in": businesses}},
                                            {"_id": 1, "stars": 1, "text": 1, "business_id": 1})
               if has_text(r)]
    reviews.sort(key=lambda r: r["_id"])  # Mongo order is not stable; the seed needs a fixed input
    picked = []
    for bucket, n in quota.items():
        pool = [r for r in reviews if _bucket(r["stars"]) == bucket]
        if len(pool) < n:
            sys.exit(f"only {len(pool)} reviews in bucket {bucket} for {businesses}, need {n}")
        picked += rng.sample(pool, n)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if SPLIT_PATH.exists() and not args.force:
        sys.exit(f"{SPLIT_PATH} exists; pass --force to rebuild it")
    rng = random.Random(SEED)
    train = _sample(TRAIN_BUSINESSES, TRAIN_QUOTA, rng)
    holdout = _sample(HOLDOUT_BUSINESSES, HOLDOUT_QUOTA, rng)
    assert not {r["business_id"] for r in train} & {r["business_id"] for r in holdout}
    out = {"seed": SEED,
           "train_businesses": TRAIN_BUSINESSES, "holdout_businesses": HOLDOUT_BUSINESSES,
           "train": [r["_id"] for r in train], "holdout": [r["_id"] for r in holdout]}
    SPLIT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, rows in (("train", train), ("holdout", holdout)):
        print(f"{name}: {len(rows)} reviews, stars {dict(sorted(Counter(r['stars'] for r in rows).items()))}, "
              f"businesses {dict(Counter(r['business_id'] for r in rows))}")


if __name__ == "__main__":
    main()
