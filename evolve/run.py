"""Evolves the reply prompt with parallel mutation agents on Luminal.

    python -m evolve.run --rounds 8 --agents 8 --k 4            # full run, then held-out eval
    python -m evolve.run --rounds 1 --agents 4 --no-holdout     # one exploratory round
    add --yes to skip the cost confirmation

Each round: take the top-k prompts by training score, give each of N agents a parent
(two for crossover), the parent's 5 worst training pairs, and one strategy (rotating
through STRATEGIES across the whole run). Every returned prompt is scored on the 40
training reviews, deduped by hash, and the population keeps the top 12.
Held-out reviews are only touched by final_eval, after the last round.
"""
import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from evolve import llm_client, score
from evolve.common import EVOLVE_DIR, PROMPTS_DIR, config, db

POPULATION_CAP = 12
AGENT_TEMPERATURE = 0.7
AGENT_MAX_TOKENS = 2000
RUNS_DIR = EVOLVE_DIR / "runs"
RESULTS_PATH = EVOLVE_DIR / "results.json"

STRATEGIES = [
    ("tighten", "Add or sharpen one or two rules that directly fix the failures shown above. "
                "Keep the rest of the prompt as it is."),
    ("exemplify", "Add one short few-shot example built from one of the failed cases above: the "
                  "input fields and the reply an excellent owner would post. If the prompt already "
                  "has examples, replace the weakest one instead. Keep the rest as it is."),
    ("persona", "Rewrite only the voice/persona part: who is writing and how they talk. Leave the "
                "rules, examples and output format unchanged. If there is no persona part, add one."),
    ("crossover", "Merge the best parts of the two parent prompts into one coherent prompt. Prefer "
                  "the higher-scoring parent where they conflict."),
    ("shorten", "Cut the prompt to about two thirds of its current length. Keep every instruction "
                "that matters for the score; drop repetition and filler."),
    ("restructure", "Reorganize the prompt into three sections in this order: rules, examples, "
                    "output format. Keep the content, remove contradictions."),
]
STRATEGY_TEXT = dict(STRATEGIES)

CHECK_TEXT = {
    "length": "reply must be 25-60 words",
    "specific": "reply must reuse a concrete word from the review (a dish, a person, a problem)",
    "name": "reply must use the reviewer's first name when the display name is a real name",
    "outcome": "1-2 stars: include the business phone number and a direct offer to fix it; "
               "4-5 stars: invite them back; 3 stars: either",
    "clean": 'never "valued", "feedback", "strive", "apologies for any inconvenience", '
             '"sorry you feel"; no emojis; at most one "!"',
}

AGENT_PROMPT = """You improve a system prompt that writes replies to Google reviews for small local businesses.

HOW THE PROMPT IS USED
It is the system message. The user message then contains exactly:
Business: <name> (<category>)
Business phone: <phone, or "none listed">
Reviewer: <display name>
Rating: <1-5>/5
Review:
<review text>
The model's entire output is posted publicly as the owner's reply.

HOW REPLIES ARE SCORED (0-100 per review, averaged)
- length (15): {length}
- specific (15): {specific}
- name (10): {name}
- outcome (15): {outcome}
- clean (10): {clean}
- judge (35): an editor rates 0-10 how much it sounds like a real local owner wrote it and how well it addresses this specific review

{parents}

THE 5 LOWEST-SCORING REPLIES FROM {worst_from}
{worst}

YOUR STRATEGY: {strategy}
{strategy_text}

Return the complete new prompt and nothing else. No commentary, no title, no code fences, no "Here is". The prompt must work for any business and any review, not just the examples above."""


def prompt_id(text):
    return "p" + hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()[:10]


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def _clean_prompt(text):
    """Strip the wrappers agents add despite instructions: code fences, 'Here is ...:' lines."""
    text = text.strip()
    fence = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"\A(?:here(?:'s| is)[^\n]*:|new prompt:?|improved prompt:?)\s*\n", "", text, flags=re.I)
    text = text.strip()
    if text.startswith("<<<") and text.endswith(">>>"):
        text = text[3:-3].strip()
    return text


def _format_worst(worst):
    rows = []
    for i, w in enumerate(worst, 1):
        failed = ", ".join(w["failed"]) or "none (lost points on judge only)"
        rows.append(f"#{i} score {w['score']:.1f} | judge {w['judge']:.0f}/10 | failed checks: {failed}\n"
                    f"Rating {w['stars']}/5, reviewer \"{w['author']}\"\n"
                    f"Review: {w['review']}\n"
                    f"Reply: {w['reply']}")
    return "\n\n".join(rows)


def build_agent_prompt(parents, strategy):
    if len(parents) == 1:
        block = f"PARENT PROMPT (training score {parents[0]['score']:.1f})\n<<<\n{parents[0]['text']}\n>>>"
    else:
        block = "\n\n".join(f"PARENT {chr(65 + i)} (training score {p['score']:.1f})\n<<<\n{p['text']}\n>>>"
                            for i, p in enumerate(parents))
    return AGENT_PROMPT.format(**CHECK_TEXT, parents=block,
                               worst_from="PARENT A" if len(parents) > 1 else "THE PARENT",
                               worst=_format_worst(parents[0]["worst"]),
                               strategy=strategy, strategy_text=STRATEGY_TEXT[strategy])


# ---------------------------------------------------------------- run state

class Run:
    def __init__(self, train_ids, run_id):
        self.train_ids = train_ids
        self.run_id = run_id
        self.dir = RUNS_DIR / run_id
        (self.dir / "prompts").mkdir(parents=True, exist_ok=True)
        self.seen = {}          # prompt_id -> entry, every prompt ever scored this run
        self.population = []    # top POPULATION_CAP entries by train score
        self.rounds = []
        self.strategy_counter = 0
        self.baseline = None
        self.rate_history = []  # check_rates of every prompt scored this run, for guard_checks
        self.lock = threading.Lock()

    def evaluate(self, text, round_no, parents, strategy):
        pid = "v0" if strategy == "baseline" else prompt_id(text)
        result = score.score_prompt(text, self.train_ids)
        entry = {"id": pid, "text": text, "score": result["mean"], "round": round_no,
                 "parent_ids": [p["id"] for p in parents], "strategy": strategy,
                 "worst": result["worst"], "per_review": result["per_review"]}
        (self.dir / "prompts" / f"{pid}.txt").write_text(text, encoding="utf-8")
        self._log_mongo(entry)
        return entry

    def _log_mongo(self, entry):
        per_review = [{"review_id": rid, "score": r["score"], "judge": r["judge"],
                       "failed": r["failed"], "reply": r["reply"]}
                      for rid, r in entry["per_review"].items()]
        try:
            db().prompt_runs.insert_one({
                "run_id": self.run_id, "round": entry["round"], "prompt_id": entry["id"],
                "parent_ids": entry["parent_ids"], "strategy": entry["strategy"],
                "train_score": entry["score"], "per_review": per_review,
                "judge_version": score.JUDGE_VERSION,
                "created_at": datetime.now(timezone.utc), "prompt_text": entry["text"]})
        except Exception as e:  # logging must not kill a run that is spending tokens
            print(f"  ! prompt_runs insert failed for {entry['id']}: {type(e).__name__}: {e}", file=sys.stderr)

    def admit(self, entries):
        self.population = sorted(self.population + entries, key=lambda e: -e["score"])[:POPULATION_CAP]

    def next_strategies(self, n):
        out = [STRATEGIES[(self.strategy_counter + i) % len(STRATEGIES)][0] for i in range(n)]
        self.strategy_counter += n
        return out

    def save(self):
        slim = lambda e: {k: v for k, v in e.items() if k != "per_review"}  # noqa: E731
        (self.dir / "state.json").write_text(json.dumps({
            "run_id": self.run_id, "judge_version": score.JUDGE_VERSION,
            "baseline": slim(self.baseline) if self.baseline else None,
            "population": [slim(e) for e in self.population],
            "rounds": self.rounds}, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, train_ids, run_id):
        """Continue a run from its state.json after the last completed round.

        Restored entries have no per_review (state.json omits it); parents only need
        text, score and worst. Every prompt file in the run counts as seen for dedupe.
        """
        run = cls(train_ids, run_id)
        state = json.loads((run.dir / "state.json").read_text(encoding="utf-8"))
        if state["judge_version"] != score.JUDGE_VERSION:
            raise SystemExit(f"run {run_id} used judge {state['judge_version']}, "
                             f"current is {score.JUDGE_VERSION}; start a new run")
        run.population = state["population"]
        run.rounds = state["rounds"]
        run.baseline = state.get("baseline") or next(e for e in run.population if e["id"] == "v0")
        run.strategy_counter = sum(len(r["candidates"]) for r in run.rounds)
        for path in (run.dir / "prompts").glob("*.txt"):
            run.seen[prompt_id(path.read_text(encoding="utf-8"))] = None
        for doc in db().prompt_runs.find({"run_id": run_id}, {"per_review.failed": 1}):
            rows = doc["per_review"]
            run.rate_history.append({k: sum(1 for r in rows if k not in r["failed"]) / len(rows)
                                     for k in score.POINTS})
        return run


def check_rates(entry):
    rows = entry["per_review"].values()
    return {k: sum(1 for r in rows if r["checks"][k]) / len(rows) for k in score.POINTS}


class BrokenCheck(SystemExit):
    pass


def guard_checks(history):
    """Stop if a check has never varied: 100% pass (or 100% fail) for every prompt scored in
    the run, v0 and earlier rounds included. A check that once discriminated and is now
    passed by every candidate is saturated (prompts learned it), not broken."""
    if len(history) < 3:
        return
    for check in score.POINTS:
        values = {r[check] for r in history}
        if values in ({1.0}, {0.0}):
            raise BrokenCheck(f"STOP: check '{check}' {'passes' if values == {1.0} else 'fails'} on "
                              f"100% of replies for all {len(history)} prompts scored in this run "
                              f"- the check is broken")


def mutate(parents, strategy, tag):
    content, _ = llm_client.chat([{"role": "user", "content": build_agent_prompt(parents, strategy)}],
                                 temperature=AGENT_TEMPERATURE, max_tokens=AGENT_MAX_TOKENS, tag=tag)
    return _clean_prompt(content)


def run_round(run, round_no, n_agents, k, check_guard=True):
    top = run.population[:k]
    assignments = []
    for i, strategy in enumerate(run.next_strategies(n_agents)):
        if strategy == "crossover":
            if len(top) < 2:
                strategy = "restructure"  # crossover needs two distinct parents
                parents = [top[i % len(top)]]
            else:
                parents = [top[i % len(top)], top[(i + 1) % len(top)]]
                parents.sort(key=lambda p: -p["score"])
        else:
            parents = [top[i % len(top)]]
        assignments.append((i, strategy, parents))

    def agent(job):
        i, strategy, parents = job
        tag = f"agent:r{round_no}:{i}:{strategy}"
        try:
            text = mutate(parents, strategy, tag)
        except Exception as e:
            return {"agent": i, "strategy": strategy, "error": f"mutation failed: {type(e).__name__}: {e}"}
        if len(text) < 20:
            return {"agent": i, "strategy": strategy, "error": f"empty/too short prompt: {text!r}"}
        pid = prompt_id(text)
        with run.lock:
            duplicate = pid in run.seen
            if not duplicate:
                run.seen[pid] = None  # claim it so a concurrent identical candidate isn't scored twice
        if duplicate:
            return {"agent": i, "strategy": strategy, "id": pid, "duplicate": True}
        try:
            entry = run.evaluate(text, round_no, parents, strategy)
        except Exception as e:
            with run.lock:
                run.seen.pop(pid, None)
            return {"agent": i, "strategy": strategy, "id": pid,
                    "error": f"scoring failed: {type(e).__name__}: {e}"}
        with run.lock:
            run.seen[pid] = entry
        return {"agent": i, "strategy": strategy, "id": pid, "entry": entry}

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        outcomes = list(pool.map(agent, assignments))

    new = [o["entry"] for o in outcomes if "entry" in o]
    run.admit(new)
    best_new = max(new, key=lambda e: e["score"]) if new else None
    summary = {
        "round": round_no,
        "best_train": run.population[0]["score"],
        "best_id": run.population[0]["id"],
        "mean_new": round(statistics.mean(e["score"] for e in new), 2) if new else None,
        "best_strategy": best_new["strategy"] if best_new else None,
        "candidates": [{"agent": o["agent"], "strategy": o["strategy"], "id": o.get("id"),
                        "parent_ids": [p["id"] for p in assignments[o["agent"]][2]],
                        "score": o["entry"]["score"] if "entry" in o else None,
                        "duplicate": o.get("duplicate", False), "error": o.get("error")}
                       for o in outcomes],
    }
    run.rounds.append(summary)
    run.save()
    mean_new = f"{summary['mean_new']:.1f}" if new else "n/a"
    best_strategy = f"{best_new['strategy']} ({best_new['score']:.1f})" if new else "n/a"
    print(f"round {round_no} | best train {summary['best_train']:.1f} ({summary['best_id']}) | "
          f"mean of new {mean_new} | best strategy {best_strategy}", flush=True)
    by_id = {e["id"]: e for e in new}
    for c in summary["candidates"]:
        status = (f"{c['score']:.1f}" if c["score"] is not None
                  else "duplicate" if c["duplicate"] else f"ERROR {c['error']}")
        rates = ""
        if c["id"] in by_id:
            r = check_rates(by_id[c["id"]])
            judge = statistics.mean(x["judge"] for x in by_id[c["id"]]["per_review"].values())
            rates = "  [" + " ".join(f"{k} {v:.0%}" for k, v in r.items()) + f" judge {judge:.1f}]"
        print(f"    agent {c['agent']} {c['strategy']:<11} parents {','.join(c['parent_ids'])} "
              f"-> {c['id'] or '-'}  {status}{rates}", flush=True)
    run.rate_history += [check_rates(e) for e in new]
    if check_guard:
        guard_checks(run.rate_history)
    return summary


# ---------------------------------------------------------------- cost guard

def estimate(rounds, agents, n_train, v0_text):
    """Token estimate from the shape of the calls; prompts are assumed to grow to ~700 tokens."""
    review_msg, reply_out = 170, 110
    judge_in, judge_out = 420, 2
    prompt_tokens_avg = 700
    agent_in = 1500 + 2 * prompt_tokens_avg  # instructions + 5 worst pairs + up to two parents
    agent_out = prompt_tokens_avg
    gen = rounds * agents * n_train
    judge = gen
    calls = rounds * agents
    v0 = len(v0_text) // 4
    tokens_in = (gen * (prompt_tokens_avg + review_msg) + judge * judge_in + calls * agent_in
                 + n_train * (v0 + review_msg + judge_in))
    tokens_out = gen * reply_out + judge * judge_out + calls * agent_out + n_train * (reply_out + judge_out)
    return {"generations": gen, "judge_calls": judge, "agent_calls": calls,
            "tokens_in": tokens_in, "tokens_out": tokens_out}


def confirm(est, assume_yes):
    total = est["tokens_in"] + est["tokens_out"]
    cost = total / 1e6 * config.LLM_COST_PER_MILLION_TOKENS
    print(f"Estimated (cache misses, upper bound): {est['generations']} generations, "
          f"{est['judge_calls']} judge calls, {est['agent_calls']} agent calls")
    print(f"  ~{est['tokens_in']:,} input + ~{est['tokens_out']:,} output = ~{total:,} tokens "
          f"(~${cost:.2f} at ${config.LLM_COST_PER_MILLION_TOKENS}/M)  model {llm_client.MODEL}")
    if assume_yes:
        print("  --yes given, starting.")
        return
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        raise SystemExit("aborted")


# ---------------------------------------------------------------- final evaluation

def final_eval(run, v0_entry):
    """The only place held-out ids are used: after optimization is finished."""
    holdout = score.load_split()["holdout"]
    best = run.population[0]
    v0_hold = score.score_prompt(v0_entry["text"], holdout)
    best_hold = v0_hold if best["id"] == "v0" else score.score_prompt(best["text"], holdout)
    (PROMPTS_DIR / "best.txt").write_text(best["text"] + "\n", encoding="utf-8")
    results = {
        "run_id": run.run_id, "judge_version": score.JUDGE_VERSION, "model": llm_client.MODEL,
        "rounds": run.rounds,
        "baseline_train": v0_entry["score"], "baseline_holdout": v0_hold["mean"],
        "best_train": best["score"], "best_holdout": best_hold["mean"],
        "best_prompt_id": best["id"], "best_strategy": best["strategy"],
        "best_parent_ids": best["parent_ids"], "best_round": best["round"],
        "baseline_prompt": v0_entry["text"], "best_prompt": best["text"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_detail": {"v0": _slim_detail(v0_hold), "best": _slim_detail(best_hold)},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nheld-out ({len(holdout)} reviews): v0 {v0_hold['mean']:.1f} -> best {best['id']} "
          f"{best_hold['mean']:.1f}   (train: {v0_entry['score']:.1f} -> {best['score']:.1f})")
    print(f"wrote {RESULTS_PATH} and {PROMPTS_DIR / 'best.txt'}")
    try:
        from evolve import report  # imported late: the report can be fixed while a run is going
        print(f"wrote {report.build(results)}")
    except Exception as e:
        print(f"  ! report.html failed ({type(e).__name__}: {e}); rerun: python -m evolve.report")
    return results


def _slim_detail(result):
    return {rid: {k: r[k] for k in ("score", "judge", "failed", "reply")}
            for rid, r in result["per_review"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--no-holdout", action="store_true", help="skip the final held-out evaluation")
    ap.add_argument("--no-check-guard", action="store_true",
                    help="don't stop when a check is 100%% pass or 100%% fail for every new candidate")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue evolve/runs/RUN_ID after its last completed round, up to --rounds")
    ap.add_argument("--stop-at", metavar="HH:MM",
                    help="local time: abort the round in progress (keeping candidates already scored) "
                         "and go to the held-out evaluation")
    args = ap.parse_args()

    timer = None
    if args.stop_at:
        hh, mm = map(int, args.stop_at.split(":"))
        now = datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        timer = threading.Timer(max(0.0, (deadline - now).total_seconds()), llm_client.ABORT.set)
        timer.daemon = True
        timer.start()
        print(f"deadline {deadline:%H:%M}: rounds stop there, then held-out evaluation")

    v0_text = (PROMPTS_DIR / "v0.txt").read_text(encoding="utf-8").strip()
    train_ids = score.load_split()["train"]

    if args.resume:
        run = Run.load(train_ids, args.resume)
        first_round = len(run.rounds) + 1
        confirm(estimate(args.rounds - first_round + 1, args.agents, len(train_ids), v0_text), args.yes)
        v0_entry = run.baseline
        print(f"run {run.run_id} resumed after round {first_round - 1}: best train "
              f"{run.population[0]['score']:.1f} ({run.population[0]['id']}), v0 train {v0_entry['score']:.1f}")
    else:
        confirm(estimate(args.rounds, args.agents, len(train_ids), v0_text), args.yes)
        run = Run(train_ids, time.strftime("%Y%m%d-%H%M%S"))
        print(f"run {run.run_id}: {args.rounds} rounds x {args.agents} agents, top-{args.k}, "
              f"judge {score.JUDGE_VERSION}, state in {run.dir}")
        v0_entry = run.evaluate(v0_text, 0, [], "baseline")
        run.seen[prompt_id(v0_text)] = v0_entry
        run.baseline = v0_entry
        run.admit([v0_entry])
        run.rate_history.append(check_rates(v0_entry))
        v0_rates = check_rates(v0_entry)
        v0_judge = statistics.mean(x["judge"] for x in v0_entry["per_review"].values())
        print(f"round 0 | v0 train {v0_entry['score']:.1f}  ["
              + " ".join(f"{k} {v:.0%}" for k, v in v0_rates.items()) + f" judge {v0_judge:.1f}]", flush=True)
        first_round = 1

    for round_no in range(first_round, args.rounds + 1):
        if llm_client.ABORT.is_set():
            print(f"deadline reached before round {round_no}; stopping rounds", flush=True)
            break
        summary = run_round(run, round_no, args.agents, args.k, check_guard=not args.no_check_guard)
        if llm_client.ABORT.is_set():
            summary["aborted_at_deadline"] = True
            run.save()
            kept = sum(1 for c in summary["candidates"] if c["score"] is not None)
            print(f"deadline reached during round {round_no}: kept {kept} fully scored candidates; "
                  f"stopping rounds", flush=True)
            break
    if timer:
        timer.cancel()
    llm_client.ABORT.clear()

    best = run.population[0]
    shutil.copyfile(run.dir / "prompts" / f"{best['id']}.txt", run.dir / "best.txt")
    t = llm_client.TOTALS
    print(f"\nbest {best['id']} ({best['strategy']}, round {best['round']}) train {best['score']:.1f}; "
          f"prompt in {run.dir / 'prompts' / (best['id'] + '.txt')}")
    print(f"LLM calls {t['calls']}, prompt tokens {t['prompt_tokens']:,}, completion tokens "
          f"{t['completion_tokens']:,}, reasoning tokens {t['reasoning_tokens']}, retries {t['retries']}")
    if not args.no_holdout:
        final_eval(run, v0_entry)


if __name__ == "__main__":
    main()
