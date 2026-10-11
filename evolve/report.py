"""Writes evolve/report.html from evolve/results.json (and validation.json + blind_pairs.json when
present): a standalone page, no external assets.

    python -m evolve.report

run.py calls build() right after the held-out evaluation.
"""
import html
import json
import re
import statistics

from evolve import score
from evolve.common import EVOLVE_DIR

REPORT_PATH = EVOLVE_DIR / "report.html"            # public: reviewer names reduced to initials
FULL_REPORT_PATH = EVOLVE_DIR / "report_full.html"  # local only (git-ignored): real names
RESULTS_PATH = EVOLVE_DIR / "results.json"
CHECKS = list(score.POINTS)

_anon = None  # set by build() while rendering the public report
# First words of display names that are ordinary words, not first names ("The Foodie").
NOT_NAMES = {"the", "mr", "mrs", "ms", "dr", "miss", "big", "little", "good", "great", "best", "real",
             "not", "and", "for", "you", "our", "food", "chef", "local", "official", "team"}


def esc(value):
    text = str(value)
    if _anon is not None:
        text = _anon(text)
    return html.escape(text, quote=True)


def _anonymizer(review_ids):
    """Replaces every reviewer's display name, and their first name on its own, with initials:
    'Firstname Lastname' -> 'F. L.', 'Firstname' -> 'F.'. Covers names quoted inside prompts and
    replies, including lowercase display names, which replies quote back capitalized."""
    names = {}
    for r in score.load_reviews(review_ids).values():
        author = (r.get("author") or "").strip()
        if not author:
            continue
        initials = " ".join(p[0].upper() + "." for p in author.split() if p[0].isalpha()) or "Reviewer"
        names[author] = initials
        first = author.split()[0]
        if (re.fullmatch(r"[A-Za-z][A-Za-z']{2,}", first)
                and first.lower() not in score.STOPWORDS | NOT_NAMES):
            names.setdefault(first.capitalize(), first[0].upper() + ".")
    if not names:
        return lambda s: s
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                         + r")(?!\w)")
    return lambda s: pattern.sub(lambda m: names[m.group(1)], s)


def _rates(detail):
    rows = list(detail.values())
    out = {k: sum(1 for r in rows if k not in r["failed"]) / len(rows) for k in CHECKS}
    out["judge"] = statistics.mean(r["judge"] for r in rows)
    return out


def _chart(results):
    """Best train score per round (round 0 = v0) and mean of that round's new candidates."""
    rounds = [{"round": 0, "best": results["baseline_train"], "mean": None, "strategy": "baseline"}]
    for r in results["rounds"]:
        rounds.append({"round": r["round"], "best": r["best_train"], "mean": r["mean_new"],
                       "strategy": r["best_strategy"] or "-", "partial": r.get("aborted_at_deadline")})
    w, h, left, right, top, bottom = 640, 260, 44, 64, 16, 34
    n = max(1, len(rounds) - 1)
    x = lambda i: left + (w - left - right) * i / n  # noqa: E731
    y = lambda v: top + (h - top - bottom) * (1 - v / 100)  # noqa: E731
    parts = []
    for g in (0, 25, 50, 75, 100):
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(g):.1f}" y2="{y(g):.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{y(g) + 4:.1f}" text-anchor="end">{g}</text>')
    for i, r in enumerate(rounds):
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{h - 12}" text-anchor="middle">'
                     f'{"v0" if r["round"] == 0 else "R" + str(r["round"])}</text>')
    mean_pts = [(x(i), y(r["mean"])) for i, r in enumerate(rounds) if r["mean"] is not None]
    if len(mean_pts) > 1:
        parts.append('<polyline class="line s2" points="'
                     + " ".join(f"{a:.1f},{b:.1f}" for a, b in mean_pts) + '"/>')
    parts.append('<polyline class="line s1" points="'
                 + " ".join(f"{x(i):.1f},{y(r['best']):.1f}" for i, r in enumerate(rounds)) + '"/>')
    for i, r in enumerate(rounds):
        label = "v0 baseline" if r["round"] == 0 else f"Round {r['round']}"
        tip = f"{label}: best train {r['best']:.1f}"
        if r["mean"] is not None:
            tip += f" · mean of new {r['mean']:.1f} · best strategy {r['strategy']}"
        if r.get("partial"):
            tip += " · stopped at deadline"
        if r["mean"] is not None:
            parts.append(f'<circle class="dot s2" cx="{x(i):.1f}" cy="{y(r["mean"]):.1f}" r="4"/>')
        parts.append(f'<circle class="dot s1" cx="{x(i):.1f}" cy="{y(r["best"]):.1f}" r="5"/>'
                     f'<rect class="hit" x="{x(i) - 18:.1f}" y="{top}" width="36" height="{h - top - bottom}" '
                     f'data-tip="{esc(tip)}"/>')
    last = rounds[-1]
    parts.append(f'<text class="direct s1t" x="{x(len(rounds) - 1) + 8:.1f}" y="{y(last["best"]) + 4:.1f}">'
                 f'{last["best"]:.1f}</text>')
    parts.append(f'<text class="direct" x="{x(0) + 8:.1f}" y="{y(rounds[0]["best"]) + 4:.1f}">'
                 f'{rounds[0]["best"]:.1f}</text>')
    svg = (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Best training score per round">'
           + "".join(parts) + "</svg>")
    table = "".join(
        f'<tr><td>{"v0" if r["round"] == 0 else r["round"]}</td><td>{r["best"]:.1f}</td>'
        f'<td>{"" if r["mean"] is None else format(r["mean"], ".1f")}</td><td>{esc(r["strategy"])}</td></tr>'
        for r in rounds)
    return svg, table


def _samples(results, reviews):
    """Per rating band, the held-out review where the best prompt gained the most over v0."""
    v0, best = results["holdout_detail"]["v0"], results["holdout_detail"]["best"]
    picks = []
    for band, test in (("1-2 stars", lambda s: s <= 2), ("3 stars", lambda s: s == 3),
                       ("4-5 stars", lambda s: s >= 4)):
        ids = [rid for rid in best if test(int(reviews[rid]["stars"]))]
        if ids:
            picks.append((band, max(ids, key=lambda rid: best[rid]["score"] - v0[rid]["score"])))
    out = []
    for band, rid in picks:
        rv = reviews[rid]
        cells = []
        for name, d in (("v0", v0[rid]), ("best", best[rid])):
            failed = ", ".join(d["failed"]) or "none"
            cells.append(f'<div class="reply"><div class="reply-head"><span class="tag">{name}</span>'
                         f'<span class="num">{d["score"]:.1f}</span></div>'
                         f'<p>{esc(d["reply"])}</p>'
                         f'<div class="meta">judge {d["judge"]:.0f}/10 · failed: {esc(failed)} · '
                         f'{len(d["reply"].split())} words</div></div>')
        out.append(f'<article class="sample"><div class="sample-head">{esc(band)} · '
                   f'{esc(rv["business"].get("name", rv["business_id"]))} · {int(rv["stars"])}/5 · '
                   f'{esc(rv.get("author") or "anonymous")}</div>'
                   f'<blockquote>{esc(rv["text"].strip())}</blockquote>'
                   f'<div class="pair">{"".join(cells)}</div></article>')
    return "".join(out)


def _candidates(results):
    rows = []
    for r in results["rounds"]:
        for c in r["candidates"]:
            status = (f"{c['score']:.1f}" if c["score"] is not None
                      else "duplicate" if c.get("duplicate") else "error")
            win = ' class="win"' if c["id"] == results["best_prompt_id"] else ""
            rows.append(f'<tr{win}><td>{r["round"]}</td><td>{esc(c["strategy"])}</td>'
                        f'<td><code>{esc(",".join(c["parent_ids"]))}</code></td>'
                        f'<td><code>{esc(c["id"] or "-")}</code></td><td class="r">{status}</td></tr>')
    return "".join(rows)


VALIDATION_PATH = EVOLVE_DIR / "validation.json"
PAIRS_PATH = EVOLVE_DIR / "blind_pairs.json"
PROMPT_LABELS = {"v0": "v0 (one line)", "rules": "rules (hand-written, not optimized)",
                 "round1": "round-1 best", "best": "final best"}


def _bars(by_band, other_label):
    rows = []
    for band in ("1-2", "3", "4-5"):
        b = by_band.get(band)
        if not b:
            continue
        n = b["best"] + b["other"] + b["tie"]
        seg = lambda count, cls: (f'<div class="seg {cls}" style="flex:{count}"></div>' if count else "")  # noqa: E731
        rows.append(f'<div class="bar-row"><div class="bar-label">{band} stars <span class="muted">({n})</span></div>'
                    f'<div class="bar">{seg(b["best"], "s1bg")}{seg(b["tie"], "tiebg")}{seg(b["other"], "s2bg")}</div>'
                    f'<div class="bar-nums">best <b>{b["best"]}</b> · {esc(other_label)} <b>{b["other"]}</b>'
                    f'{f" · tie {b["tie"]}" if b["tie"] else ""}</div></div>')
    return "".join(rows)


def _validation_section():
    if not VALIDATION_PATH.exists():
        return "", ""
    v = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    s, vs = v["scores"], v["best_vs"]
    rows = ""
    for name in ("v0", "rules", "round1", "best"):
        diff = ""
        if name in vs:
            p = vs[name]
            diff = (f'{p["mean_diff"]:+.1f}</td><td class="r">[{p["ci95"][0]:+.1f}, {p["ci95"][1]:+.1f}]</td>'
                    f'<td class="r">{p["wins"]}/{p["ties"]}/{p["losses"]}')
        else:
            diff = '</td><td class="r"></td><td class="r">'
        rows += (f'<tr><td>{esc(PROMPT_LABELS[name])}</td><td class="r">{s[name]["mean"]:.1f}</td>'
                 f'<td class="r">{s[name]["judge_mean"]:.1f}</td><td class="r">{diff}</td></tr>')
    fresh = (f'<section class="card"><h2>Fresh-data check: {len(v["fresh_review_ids"])} reviews from '
             f'{len(v["fresh_businesses"])} businesses the run never saw</h2>'
             f'<p class="note">Same scorer as the run. "Best minus" columns are paired per review; CI is a 95% '
             f'bootstrap interval; W/T/L counts reviews where the final best scored higher / equal / lower.</p>'
             f'<div class="scroll"><table><tr><th>Prompt</th><th class="r">Score</th><th class="r">Judge /10</th>'
             f'<th class="r">Best minus</th><th class="r">95% CI</th><th class="r">W/T/L</th></tr>{rows}</table></div>'
             f'</section>')

    blind = v.get("blind_judge")
    if not blind:
        return fresh, ""
    t = blind["tally"]
    blocks = []
    for other in ("rules", "round1"):
        if other not in t:
            continue
        x = t[other]
        blocks.append(f'<div class="cmp"><h3>Final best vs {esc(PROMPT_LABELS[other])}</h3>'
                      f'<p class="cmp-total">best <b>{x["best"]}</b> – <b>{x["other"]}</b> {esc(other)}'
                      f'{f", {x["tie"]} ties" if x["tie"] else ""} · sign test p = {x["sign_test_p"]:g} · '
                      f'reply shown first won {x["A_won"]}/{x["best"] + x["other"]}</p>'
                      f'{_bars(x["by_band"], other)}</div>')
    examples = ""
    if PAIRS_PATH.exists():
        pairs = {p["pair_id"]: p for p in json.loads(PAIRS_PATH.read_text(encoding="utf-8"))}
        lost = [(pairs[vote["pair_id"]], vote) for vote in blind["votes"]
                if vote["pair_id"] in pairs and pairs[vote["pair_id"]]["comparison"] == "round1"
                and pairs[vote["pair_id"]]["stars"] <= 2 and vote["winner"] != "tie"
                and pairs[vote["pair_id"]][f'{vote["winner"]}_is'] == "round1"][:3]
        for p, vote in lost:
            best_reply = p["A"] if p["A_is"] == "best" else p["B"]
            r1_reply = p["B"] if p["A_is"] == "best" else p["A"]
            examples += (f'<article class="sample"><div class="sample-head">{esc(p["business"])} · '
                         f'{int(p["stars"])}/5 · {esc(p["author"])}</div>'
                         f'<blockquote>{esc(p["review"])}</blockquote><div class="pair">'
                         f'<div class="reply"><div class="reply-head"><span class="tag">final best</span>'
                         f'<span class="lost">lost</span></div><p>{esc(best_reply)}</p></div>'
                         f'<div class="reply"><div class="reply-head"><span class="tag">round-1 best</span>'
                         f'<span class="won">preferred</span></div><p>{esc(r1_reply)}</p></div></div>'
                         f'<div class="meta">Judge: {esc(vote["reason"])}</div></article>')
    section = (f'<section class="card"><h2>Independent blind judge</h2>'
               f'<p class="note">{esc(blind["judge"])}. Each judge saw one review and two replies labelled A and B, '
               f'with no rubric and no idea which prompt wrote which, and picked the reply the owner should post.</p>'
               f'<div class="legend"><span style="--c: var(--s1)">final best wins</span>'
               f'<span style="--c: var(--tie)">tie</span><span style="--c: var(--s2)">other prompt wins</span></div>'
               f'<div class="grid2">{"".join(blocks)}</div>'
               + (f'<h3 class="sub-h">Complaints where the round-1 prompt was preferred</h3>{examples}' if examples else "")
               + '</section>')
    return fresh, section


def _verdict():
    if not VALIDATION_PATH.exists():
        return ""
    v = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    vs, t = v["best_vs"], (v.get("blind_judge") or {}).get("tally", {})
    items = [f'<li><b>Writing the rules down is most of the gain.</b> On fresh reviews, v0 scores '
             f'{v["scores"]["v0"]["mean"]:.1f} and the hand-written rules prompt {v["scores"]["rules"]["mean"]:.1f}.</li>']
    if "rules" in t:
        items.append(f'<li><b>The loop beats the hand-written rules.</b> {vs["rules"]["mean_diff"]:+.1f} points on our '
                     f'scorer (CI {vs["rules"]["ci95"][0]:+.1f} to {vs["rules"]["ci95"][1]:+.1f}), and an independent '
                     f'blind judge preferred it {t["rules"]["best"]}–{t["rules"]["other"]} (p = {t["rules"]["sign_test_p"]:g}).</li>')
    if "round1" in t:
        b = t["round1"]["by_band"]
        items.append(f'<li><b>Rounds 2–5 did not help, and hurt complaints.</b> Our scorer: {vs["round1"]["mean_diff"]:+.1f} '
                     f'(CI {vs["round1"]["ci95"][0]:+.1f} to {vs["round1"]["ci95"][1]:+.1f}, not significant). Blind judge: '
                     f'{t["round1"]["best"]}–{t["round1"]["other"]} overall, but the round-1 prompt won '
                     f'{b.get("1-2", {}).get("other", 0)} of {sum(b.get("1-2", {}).values())} complaints and '
                     f'{b.get("3", {}).get("other", 0)} of {sum(b.get("3", {}).values())} 3-star reviews, while the final '
                     f'prompt won {b.get("4-5", {}).get("best", 0)} of {sum(b.get("4-5", {}).values())} 4–5 star reviews.</li>')
    return (f'<section class="card verdict"><h2>Verdict</h2><ul>{"".join(items)}</ul>'
            f'<p class="rec"><b>Recommendation:</b> use the round-1 prompt (<code>{esc(v["round1_prompt_id"])}</code>) '
            f'for 1–3 star reviews and the final prompt for 4–5 stars. Both prompts embed real training reviews in '
            f'their examples (reviewer names, review text, business phone numbers): swap those for invented '
            f'placeholders before production. Add a pairwise judge to the loop before running it again. '
            f'A human blind test is still the real proof.</p></section>')


def build(results=None):
    """Writes report_full.html (real names, local) and report.html (initials, public); returns the latter."""
    global _anon
    if results is None:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    split = score.load_split()
    ids = split["train"] + split["holdout"]
    if VALIDATION_PATH.exists():
        ids += json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))["fresh_review_ids"]
    try:
        _anon = None
        FULL_REPORT_PATH.write_text(_render(results), encoding="utf-8")
        _anon = _anonymizer(ids)
        REPORT_PATH.write_text(_render(results), encoding="utf-8")
    finally:
        _anon = None
    return REPORT_PATH


def _render(results):
    holdout_ids = list(results["holdout_detail"]["best"])
    reviews = score.load_reviews(holdout_ids)
    rv0, rbest = _rates(results["holdout_detail"]["v0"]), _rates(results["holdout_detail"]["best"])
    svg, round_table = _chart(results)
    delta = results["best_holdout"] - results["baseline_holdout"]
    stopped = any(r.get("aborted_at_deadline") for r in results["rounds"])
    check_rows = "".join(
        f'<tr><td>{k}</td><td class="r">{score.POINTS[k]}</td><td class="r">{rv0[k]:.0%}</td>'
        f'<td class="r">{rbest[k]:.0%}</td></tr>' for k in CHECKS)
    check_rows += (f'<tr><td>judge (0-10, ×3.5)</td><td class="r">35</td><td class="r">{rv0["judge"]:.1f}</td>'
                   f'<td class="r">{rbest["judge"]:.1f}</td></tr>')
    fresh, blind = _validation_section()
    extra_prompts = ""
    if VALIDATION_PATH.exists():
        r1_id = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))["round1_prompt_id"]
        r1_path = EVOLVE_DIR / "runs" / results["run_id"] / "prompts" / f"{r1_id}.txt"
        rules_path = EVOLVE_DIR / "prompts" / "rules.txt"
        cols = []
        if rules_path.exists():
            cols.append(f'<div><div class="col-head">rules (hand-written baseline)</div>'
                        f'<pre>{esc(rules_path.read_text(encoding="utf-8").strip())}</pre></div>')
        if r1_path.exists():
            cols.append(f'<div><div class="col-head">round-1 best · {esc(r1_id)}</div>'
                        f'<pre>{esc(r1_path.read_text(encoding="utf-8").strip())}</pre></div>')
        if cols:
            extra_prompts = (f'<details class="more"><summary>Rules baseline and round-1 prompt</summary>'
                             f'<div class="grid2">{"".join(cols)}</div></details>')
    page = TEMPLATE.format(
        verdict=_verdict(), fresh=fresh, blind=blind, extra_prompts=extra_prompts,
        privacy=" Reviewer names are shown as initials, including inside prompts and replies." if _anon else "",
        run_id=esc(results["run_id"]), model=esc(results["model"]), judge=esc(results["judge_version"]),
        evaluated=esc(results.get("evaluated_at", "")), n_rounds=len(results["rounds"]),
        stopped=" The last round was cut short at the deadline; only fully scored candidates were kept."
        if stopped else "",
        v0_hold=f'{results["baseline_holdout"]:.1f}', best_hold=f'{results["best_holdout"]:.1f}',
        delta=f"{delta:+.1f}", v0_train=f'{results["baseline_train"]:.1f}',
        best_train=f'{results["best_train"]:.1f}', best_id=esc(results["best_prompt_id"]),
        best_strategy=esc(results.get("best_strategy", "")), best_round=esc(results.get("best_round", "")),
        n_hold=len(holdout_ids), chart=svg, round_table=round_table, check_rows=check_rows,
        v0_prompt=esc(results["baseline_prompt"]), best_prompt=esc(results["best_prompt"]),
        samples=_samples(results, reviews), candidates=_candidates(results))
    return page


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt Evolution Report</title>
<style>
:root {{
  --page: #f9f9f7; --surface: #fcfcfb; --border: #e4e3de; --text: #0b0b0b; --text-2: #52514e;
  --muted: #8a8984; --s1: #2a78d6; --s2: #eb6834; --good: #0ca30c; --win: #eef4fc; --tie: #c9c8c2;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --page: #0d0d0d; --surface: #1a1a19; --border: #383835; --text: #f0efec; --text-2: #c3c2b7;
    --muted: #8f8e88; --s1: #3987e5; --s2: #d95926; --good: #0ca30c; --win: #1d2a3a; --tie: #5a5955; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --page: #0d0d0d; --surface: #1a1a19; --border: #383835; --text: #f0efec; --text-2: #c3c2b7;
  --muted: #8f8e88; --s1: #3987e5; --s2: #d95926; --good: #0ca30c; --win: #1d2a3a; --tie: #5a5955; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
.sub {{ color: var(--text-2); margin: 0 0 24px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; margin-bottom: 20px; }}
.heroes {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
.hero .label {{ color: var(--text-2); font-size: 13px; }}
.hero .big {{ font-size: 44px; font-weight: 650; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.hero .small {{ color: var(--text-2); font-size: 13px; font-variant-numeric: tabular-nums; }}
.delta {{ color: var(--good); }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.grid {{ stroke: var(--border); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.line.s1 {{ stroke: var(--s1); }} .line.s2 {{ stroke: var(--s2); }}
.dot {{ stroke: var(--surface); stroke-width: 2; }}
.dot.s1 {{ fill: var(--s1); }} .dot.s2 {{ fill: var(--s2); }}
.direct {{ fill: var(--text-2); font-size: 12px; font-variant-numeric: tabular-nums; }}
.s1t {{ fill: var(--text); font-weight: 600; }}
.hit {{ fill: transparent; cursor: crosshair; }}
.legend {{ display: flex; gap: 18px; color: var(--text-2); font-size: 13px; margin: 0 0 8px; }}
.legend span::before {{ content: ""; display: inline-block; width: 14px; height: 2px; vertical-align: middle;
  margin-right: 6px; background: var(--c); }}
#tip {{ position: fixed; pointer-events: none; background: var(--text); color: var(--page); font-size: 12px;
  padding: 6px 8px; border-radius: 6px; display: none; max-width: 300px; z-index: 5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-2); font-weight: 600; font-size: 12px; }}
td.r, th.r {{ text-align: right; }}
tr.win td {{ background: var(--win); font-weight: 600; }}
code {{ font-size: 12px; }}
pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font: 13px/1.5 ui-monospace, Consolas, monospace;
  background: var(--page); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
.col-head {{ font-size: 13px; color: var(--text-2); margin-bottom: 6px; }}
.sample {{ border-top: 1px solid var(--border); padding-top: 16px; margin-top: 16px; }}
.sample:first-of-type {{ border-top: 0; margin-top: 0; padding-top: 0; }}
.sample-head {{ font-size: 13px; color: var(--text-2); }}
blockquote {{ margin: 8px 0 12px; padding-left: 12px; border-left: 3px solid var(--border); color: var(--text-2); }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.reply {{ border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
.reply p {{ margin: 6px 0; white-space: pre-wrap; }}
.reply-head {{ display: flex; justify-content: space-between; align-items: center; }}
.tag {{ font-size: 12px; font-weight: 600; color: var(--text-2); text-transform: uppercase; letter-spacing: .04em; }}
.num {{ font-weight: 650; font-variant-numeric: tabular-nums; }}
.meta {{ font-size: 12px; color: var(--muted); }}
details summary {{ cursor: pointer; color: var(--text-2); }}
.note {{ color: var(--text-2); font-size: 13px; }}
.scroll {{ overflow-x: auto; }}
h3 {{ font-size: 15px; margin: 0 0 6px; }}
.sub-h {{ margin-top: 24px; }}
.verdict ul {{ margin: 0 0 12px; padding-left: 20px; }}
.verdict li {{ margin-bottom: 6px; }}
.rec {{ margin: 0; padding: 10px 12px; background: var(--page); border-radius: 8px; border: 1px solid var(--border); }}
.cmp-total {{ margin: 0 0 10px; color: var(--text-2); font-size: 13px; font-variant-numeric: tabular-nums; }}
.bar-row {{ margin-bottom: 10px; }}
.bar-label {{ font-size: 13px; margin-bottom: 3px; }}
.muted {{ color: var(--muted); }}
.bar {{ display: flex; gap: 2px; height: 14px; }}
.seg {{ border-radius: 3px; min-width: 4px; }}
.s1bg {{ background: var(--s1); }} .s2bg {{ background: var(--s2); }} .tiebg {{ background: var(--tie); }}
.bar-nums {{ font-size: 12px; color: var(--text-2); margin-top: 3px; font-variant-numeric: tabular-nums; }}
.won, .lost {{ font-size: 12px; font-weight: 600; }}
.won {{ color: var(--good); }} .lost {{ color: var(--muted); }}
details.more {{ margin-top: 14px; }}
details.more .grid2 {{ margin-top: 10px; }}
@media (max-width: 720px) {{
  .heroes, .grid2, .pair {{ grid-template-columns: 1fr; }}
  .hero .big {{ font-size: 36px; }}
}}
</style>
</head>
<body>
<main>
  <h1>Prompt evolution</h1>
  <p class="sub">Run {run_id} · {n_rounds} rounds · model {model} · judge {judge} · evaluated {evaluated}.{stopped}{privacy}</p>

  {verdict}

  <section class="card heroes">
    <div class="hero"><div class="label">v0 baseline · held-out</div><div class="big">{v0_hold}</div>
      <div class="small">train {v0_train}</div></div>
    <div class="hero"><div class="label">best prompt · held-out</div><div class="big">{best_hold}</div>
      <div class="small">train {best_train} · <code>{best_id}</code> ({best_strategy}, round {best_round})</div></div>
    <div class="hero"><div class="label">held-out gain</div><div class="big delta">{delta}</div>
      <div class="small">{n_hold} reviews from 5 businesses never seen in training</div></div>
  </section>

  <section class="card">
    <h2>Best training score per round</h2>
    <div class="legend"><span style="--c: var(--s1)">best in population</span>
      <span style="--c: var(--s2)">mean of that round's new candidates</span></div>
    {chart}
    <details><summary>Table view</summary><div class="scroll"><table>
      <tr><th>Round</th><th>Best train</th><th>Mean of new</th><th>Best strategy</th></tr>{round_table}
    </table></div></details>
  </section>

  {fresh}

  {blind}

  <section class="card">
    <h2>Held-out pass rates by check</h2>
    <div class="scroll"><table>
      <tr><th>Check</th><th class="r">Points</th><th class="r">v0</th><th class="r">best</th></tr>{check_rows}
    </table></div>
  </section>

  <section class="card">
    <h2>Prompts side by side</h2>
    <div class="grid2">
      <div><div class="col-head">v0</div><pre>{v0_prompt}</pre></div>
      <div><div class="col-head">best · {best_id}</div><pre>{best_prompt}</pre></div>
    </div>
    {extra_prompts}
  </section>

  <section class="card">
    <h2>Same held-out review, both replies</h2>
    <p class="note">For each rating band, the held-out review where the best prompt gained the most over v0.</p>
    {samples}
  </section>

  <section class="card">
    <h2>Every candidate</h2>
    <div class="scroll"><table>
      <tr><th>Round</th><th>Strategy</th><th>Parents</th><th>Prompt</th><th class="r">Train</th></tr>{candidates}
    </table></div>
    <p class="note">Scores are means over the 40 training reviews (0-100: 65 deterministic points + 35 from an
      LLM judge). Held-out numbers use 30 reviews from different businesses and were computed once, after the
      last round. With 30 reviews, differences of a few points are within noise.</p>
  </section>
</main>
<div id="tip"></div>
<script>
const tip = document.getElementById("tip");
document.querySelectorAll(".hit").forEach(el => {{
  el.addEventListener("mousemove", e => {{
    tip.textContent = el.dataset.tip; tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 310) + "px"; tip.style.top = (e.clientY + 12) + "px";
  }});
  el.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});
}});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"wrote {build()}")
