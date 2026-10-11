"""Un-blinds the independent judge's votes and adds the tally to evolve/validation.json.

    python -m evolve.blind_tally votes.json

votes.json: [{"pair_id", "winner": "A"|"B"|"tie", "reason"}] from judges that saw only the
review and replies A/B (randomized order), never which prompt wrote which.
"""
import json
import math
import sys
from pathlib import Path

from evolve.common import EVOLVE_DIR

PAIRS = EVOLVE_DIR / "blind_pairs.json"
OUT = EVOLVE_DIR / "validation.json"


def sign_test_p(wins, losses):
    """Two-sided exact sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    votes = {v["pair_id"]: v for v in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))}
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    tally = {}
    for p in pairs:
        t = tally.setdefault(p["comparison"], {"best": 0, "other": 0, "tie": 0, "missing": 0,
                                               "best_was_A": 0, "A_won": 0, "by_band": {}})
        v = votes.get(p["pair_id"])
        band = "1-2" if p["stars"] <= 2 else "3" if p["stars"] == 3 else "4-5"
        b = t["by_band"].setdefault(band, {"best": 0, "other": 0, "tie": 0})
        t["best_was_A"] += p["A_is"] == "best"
        if v is None:
            t["missing"] += 1
            continue
        if v["winner"] == "tie":
            t["tie"] += 1
            b["tie"] += 1
            continue
        t["A_won"] += v["winner"] == "A"
        who = "best" if p[f"{v['winner']}_is"] == "best" else "other"
        t[who] += 1
        b[who] += 1
    for name, t in tally.items():
        t["sign_test_p"] = round(sign_test_p(t["best"], t["other"]), 4)
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data["blind_judge"] = {"judge": "Claude subagents (different model family from generator/judge), "
                                    "pairwise, randomized A/B, no rubric", "tally": tally,
                           "votes": list(votes.values())}
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, t in tally.items():
        print(f"best vs {name}: best {t['best']}, {name} {t['other']}, tie {t['tie']}, missing {t['missing']} "
              f"| sign test p={t['sign_test_p']} | A won {t['A_won']}/{t['best'] + t['other']} (position check) "
              f"| by stars {t['by_band']}")


if __name__ == "__main__":
    main()
