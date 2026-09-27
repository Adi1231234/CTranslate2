"""Summary of a ladder_recorder log (seeded.py LADDER_LOG): per window, each attempt's temperature, token counts
(the kept hypothesis first, then the longest), whether a hypothesis ran to the token budget, compression ratio,
avg log-prob, pass or fail, and seconds; then totals: time in attempts that reached the budget and in the others.
The budget (`whisper.cc`): the prompt's last token starts the decode, so 448 - (prompt - 1) steps are left, or
224 with the capped build. Measured 27.9 with the 3-token prompt: at the full budget (446) a beam returns 444
tokens and a sample 444-446, so a hypothesis within 2 tokens of the budget counts as reaching it.
usage: ladder_report.py <ladder log> <budget: capped | full> [--brief]"""
import json, sys

LOG, BUDGET = sys.argv[1], sys.argv[2]
brief = "--brief" in sys.argv


def loop_start(tokens, max_period=150):
    """The earliest step from which the tokens repeat with one period, and that period (tokens[i] == tokens[i + p]
    for every i >= step): where a repetition loop begins."""
    best = (len(tokens), 0)
    for p in range(1, min(max_period, len(tokens) // 2) + 1):
        s = len(tokens) - p
        while s > 0 and tokens[s - 1] == tokens[s - 1 + p]:
            s -= 1
        if len(tokens) - s >= 2 * p:                            # at least two whole periods
            best = min(best, (s, p))
    return best


def budget(prompt):
    left = 448 - (prompt - 1)
    return min(224, left) if BUDGET == "capped" else left


tot = {"clips": 0, "windows": 0, "attempts": 0, "secs": 0.0, "at_budget": 0, "secs_at_budget": 0.0,
       "cr_fail": 0, "lp_fail": 0, "all_failed_windows": 0}
for line in open(LOG, encoding="utf-8"):
    clip = json.loads(line)
    tot["clips"] += 1
    if not brief:
        print(f"## {clip['unit']} {clip['uuid']}")
    for w, win in enumerate(clip["windows"]):
        tot["windows"] += 1
        atts = win["attempts"]
        tot["all_failed_windows"] += not any(a["passes"] for a in atts)
        for a in atts:
            full = max(a["n"]) >= budget(a["prompt"]) - 2
            tot["attempts"] += 1; tot["secs"] += a["secs"]
            tot["at_budget"] += full; tot["secs_at_budget"] += a["secs"] * full
            tot["cr_fail"] += a["cr"] > 2.4; tot["lp_fail"] += a["avg_lp"] < -1.0
            if not brief:
                print(f"  w{w} T{a['T']:.1f} n {a['n'][0]:3d}/{max(a['n']):3d} of {budget(a['prompt'])}"
                      f"{' FULL' if full else '     '} cr {a['cr']:.2f} lp {a['avg_lp']:6.3f} nsp {a['nsp']:.2f}"
                      f" {'pass' if a['passes'] else 'fail'} {a['secs']:.2f}s"
                      + (" loop from step {} (period {})".format(*loop_start(a["tokens"])) if full else ""))
        if not brief:
            r = win["returned"]
            print(f"  w{w} -> T{r['T']:.1f} cr {r['cr']:.2f} lp {r['avg_lp']:.3f} n {len(r['tokens'])}")
tot["secs"] = round(tot["secs"], 1); tot["secs_at_budget"] = round(tot["secs_at_budget"], 1)
print(json.dumps(tot))
