"""Validation on reviews nobody has seen: businesses scraped after split.json was made.

    python -m evolve.validate

Scores four prompts on 40 fresh reviews (stratified by stars, fixed seed):
    v0      the one-line baseline
    rules   the scoring rules written down once by hand (prompts/rules.txt), never optimized
    round1  the best prompt after round 1 of the run
    best    the final best prompt
Writes evolve/validation.json (scores, paired stats vs best) and evolve/blind_pairs.json
(best vs rules and best vs round1, A/B order randomized) for an independent blind judge.
"""
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor

from evolve import score
from evolve.common import EVOLVE_DIR, PROMPTS_DIR, db, has_text

SEED = 20261012
QUOTA = {"low": 14, 3: 6, 4: 6, 5: 14}
RUN_DIR = EVOLVE_DIR / "runs" / "20261011-170151"
OUT = EVOLVE_DIR / "validation.json"
PAIRS = EVOLVE_DIR / "blind_pairs.json"


def fresh_ids():
    split = score.load_split()
    used = split["train_businesses"] + split["holdout_businesses"]
    rows = sorted((r for r in db().reviews.find({"business_id": {"$nin": used}},
                                                {"stars": 1, "text": 1, "business_id": 1})
                   if has_text(r)), key=lambda r: r["_id"])
    rng = random.Random(SEED)
    picked = []
    for bucket, n in QUOTA.items():
        pool = [r for r in rows if ("low" if r["stars"] <= 2 else r["stars"]) == bucket]
        picked += rng.sample(pool, n)
    return [r["_id"] for r in picked], sorted({r["business_id"] for r in picked})


def paired(a, b, ids):
    diff = [a[i]["score"] - b[i]["score"] for i in ids]
    rng = random.Random(0)
    boot = sorted(statistics.mean(rng.choices(diff, k=len(diff))) for _ in range(10000))
    return {"mean_diff": round(statistics.mean(diff), 2), "ci95": [round(boot[250], 2), round(boot[9750], 2)],
            "wins": sum(d > 0 for d in diff), "ties": sum(d == 0 for d in diff), "losses": sum(d < 0 for d in diff)}


def main():
    ids, businesses = fresh_ids()
    results = json.loads((EVOLVE_DIR / "results.json").read_text(encoding="utf-8"))
    state = json.loads((RUN_DIR / "state.json").read_text(encoding="utf-8"))
    prompts = {
        "v0": (PROMPTS_DIR / "v0.txt").read_text(encoding="utf-8").strip(),
        "rules": (PROMPTS_DIR / "rules.txt").read_text(encoding="utf-8").strip(),
        "round1": (RUN_DIR / "prompts" / f"{state['rounds'][0]['best_id']}.txt").read_text(encoding="utf-8").strip(),
        "best": results["best_prompt"].strip(),
    }
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        scored = dict(zip(prompts, pool.map(lambda name: score.score_prompt(prompts[name], ids), prompts)))
    per = {name: scored[name]["per_review"] for name in prompts}
    summary = {}
    for name in prompts:
        rows = per[name].values()
        summary[name] = {"mean": scored[name]["mean"],
                         "judge_mean": round(statistics.mean(r["judge"] for r in rows), 2),
                         "pass_rates": {k: round(sum(1 for r in rows if r["checks"][k]) / len(ids), 3)
                                        for k in score.POINTS}}
    out = {"fresh_review_ids": ids, "fresh_businesses": businesses, "seed": SEED,
           "round1_prompt_id": state["rounds"][0]["best_id"], "best_prompt_id": results["best_prompt_id"],
           "scores": summary,
           "best_vs": {name: paired(per["best"], per[name], ids) for name in ("v0", "rules", "round1")},
           "replies": {name: {i: per[name][i]["reply"] for i in ids} for name in prompts}}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    reviews = score.load_reviews(ids)
    rng = random.Random(SEED + 1)
    pairs = []
    for other in ("rules", "round1"):
        for i in ids:
            best_first = rng.random() < 0.5
            a, b = ("best", other) if best_first else (other, "best")
            rv = reviews[i]
            pairs.append({"pair_id": f"{other}-{len(pairs)}", "comparison": other, "review_id": i,
                          "business": rv["business"].get("name", rv["business_id"]),
                          "category": rv["business"].get("category", ""), "stars": rv["stars"],
                          "author": rv.get("author") or "anonymous", "review": rv["text"].strip(),
                          "A": per[a][i]["reply"], "B": per[b][i]["reply"], "A_is": a, "B_is": b})
    PAIRS.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"fresh set: {len(ids)} reviews from {len(businesses)} businesses never used in the run")
    for name, s in summary.items():
        print(f"  {name:<7} {s['mean']:5.1f}  judge {s['judge_mean']:.1f}  "
              + " ".join(f"{k} {v:.0%}" for k, v in s["pass_rates"].items()))
    for name, p in out["best_vs"].items():
        print(f"  best - {name:<7} {p['mean_diff']:+.1f}  CI95 [{p['ci95'][0]:+.1f}, {p['ci95'][1]:+.1f}]  "
              f"W/T/L {p['wins']}/{p['ties']}/{p['losses']}")
    print(f"wrote {OUT} and {PAIRS} ({len(pairs)} blind pairs)")


if __name__ == "__main__":
    main()
