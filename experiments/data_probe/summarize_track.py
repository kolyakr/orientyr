"""Summarize track.json (tracking with a prior) next to stress.json (global search, no prior) as markdown tables.

Usage: python summarize_track.py <out dir containing track.json and stress.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
d = json.loads((out / "track.json").read_text())
stress = json.loads((out / "stress.json").read_text())
trials, ACC = d["trials"], d["accept"]
PRI = list(d["priors"])
N = 17


def accepted(r, m):
    return r["e"] is not None and (ACC.get(m) is None or (r["conf"] or 0) >= ACC[m])


def cascade(t, p, order):
    """First accepted method in `order`; 'global' is the no-prior fallback (relocalization)."""
    for m in order:
        r = t["global"] if m == "global" else t[p][m]
        if accepted(r, m):
            return r["e"]
    return None


def score(es):
    ok = sum(1 for e in es if e is not None and e < 5)
    wr = sum(1 for e in es if e is not None and e > 20)
    return ok, wr


def cell(ok, wr):
    return f"{ok}" + (f" ({wr}✗)" if wr else "")


conds = [(k, l) for k in ["tilt", "blur", "motion", "noise_jpeg", "moire", "photo_glare", "phone_combo"] for l in range(3)]
ORDER = ["win_sift", "ncc_ecc", "global"]

print("## Success out of 17 (wrong fixes in brackets)\n")
hdr = "| condition | global, no prior | " + " | ".join(
    f"{p}: win_sift | {p}: ncc→ecc | {p}: cascade" for p in PRI) + " |"
print(hdr)
print("|" + "---|" * (2 + 3 * len(PRI)))
tot = {}
for k, l in conds:
    sel = [t for t in trials if t["kind"] == k and t["lvl"] == l]
    g = score([t["global"]["e"] if accepted(t["global"], "global") else None for t in sel])
    s_ref = stress[f"{k}[{l}]"]
    assert round(g[0] / N, 2) == s_ref["success"], (k, l, g, s_ref)  # same images as stress.py
    row = [f"{k} L{l}", cell(*g)]
    tot.setdefault("global", [0, 0]); tot["global"][0] += g[0]; tot["global"][1] += g[1]
    for p in PRI:
        for name, es in (("win_sift", [t[p]["win_sift"]["e"] if accepted(t[p]["win_sift"], "win_sift") else None for t in sel]),
                         ("ncc_ecc", [t[p]["ncc_ecc"]["e"] if accepted(t[p]["ncc_ecc"], "ncc_ecc") else None for t in sel]),
                         ("cascade", [cascade(t, p, ORDER) for t in sel])):
            ok, wr = score(es)
            tot.setdefault((p, name), [0, 0]); tot[(p, name)][0] += ok; tot[(p, name)][1] += wr
            row.append(cell(ok, wr))
    print("| " + " | ".join(row) + " |")
n_all = N * len(conds)
print("| **all 357** | " + cell(*tot["global"]) + " | " + " | ".join(
    cell(*tot[(p, m)]) for p in PRI for m in ("win_sift", "ncc_ecc", "cascade")) + " |")

print("\n## Every method, every prior: success / wrong fixes over all 357 trials, and over the 6 hard conditions\n")
hard = [("tilt", 2), ("blur", 1), ("blur", 2), ("motion", 2), ("phone_combo", 1), ("phone_combo", 2)]
print("| prior | method | all: success | all: wrong | hard (102): success | hard: wrong | found <5 px, ignoring the accept test (hard) | median ms |")
print("|---|---|---|---|---|---|---|---|")
hsel = [t for t in trials if (t["kind"], t["lvl"]) in hard]
g_all = score([t["global"]["e"] if accepted(t["global"], "global") else None for t in trials])
g_h = score([t["global"]["e"] if accepted(t["global"], "global") else None for t in hsel])
g_f = sum(1 for t in hsel if t["global"]["e"] is not None and t["global"]["e"] < 5)
print(f"| none | global | {g_all[0]} | {g_all[1]} | {g_h[0]} | {g_h[1]} | {g_f} | {np.median([t['global']['ms'] for t in trials]):.0f} |")
for p in PRI:
    for m in ("prior", "win_sift", "ncc", "ecc", "ncc_ecc"):
        a = score([t[p][m]["e"] if accepted(t[p][m], m) else None for t in trials])
        h = score([t[p][m]["e"] if accepted(t[p][m], m) else None for t in hsel])
        f = sum(1 for t in hsel if t[p][m]["e"] is not None and t[p][m]["e"] < 5)
        ms = np.median([t[p][m]["ms"] for t in trials])
        print(f"| {p} | {m} | {a[0]} | {a[1]} | {h[0]} | {h[1]} | {f} | {ms:.0f} |")
    for name, order in (("cascade win_sift→ncc_ecc→global", ORDER), ("cascade ncc_ecc→win_sift→global", ["ncc_ecc", "win_sift", "global"])):
        a = score([cascade(t, p, order) for t in trials])
        h = score([cascade(t, p, order) for t in hsel])
        print(f"| {p} | {name} | {a[0]} | {a[1]} | {h[0]} | {h[1]} | – | – |")

print("\n## Accept tests: confidence of correct (<5 px) vs wrong (>20 px) answers, all trials and priors\n")
for m in ("global", "win_sift", "ncc", "ecc", "ncc_ecc"):
    rs = [t["global"] for t in trials] if m == "global" else [t[p][m] for t in trials for p in PRI]
    good = np.array([r["conf"] for r in rs if r["e"] is not None and r["e"] < 5], float)
    bad = np.array([r["conf"] for r in rs if r["e"] is not None and r["e"] > 20], float)
    q = lambda a, x: f"{np.percentile(a, x):.2f}" if len(a) else "–"
    zero_wrong = (bad.max() if len(bad) else float("nan"))
    print(f"- {m}: correct n={len(good)} conf p5/p50 = {q(good, 5)}/{q(good, 50)}; wrong n={len(bad)} conf p50/p95/max = "
          f"{q(bad, 50)}/{q(bad, 95)}/{zero_wrong:.2f}; declared threshold {ACC.get(m)}; correct kept at it: "
          f"{(good >= ACC[m]).sum() if m in ACC else len(good)}/{len(good)}")

print("\n## Agreement check: win_sift and ncc_ecc both accepted, but their errors differ by more than 5 px (so the fixes are at least 5 px apart)\n")
for p in PRI:
    both = [t for t in trials if accepted(t[p]["win_sift"], "win_sift") and accepted(t[p]["ncc_ecc"], "ncc_ecc")]
    dis = [t for t in both if abs(t[p]["win_sift"]["e"] - t[p]["ncc_ecc"]["e"]) > 5]
    print(f"- {p}: both accepted {len(both)}, disagree {len(dis)}")

print("\n## Accuracy when correct (median error, px)\n")
for p in PRI:
    for m in ("win_sift", "ncc", "ncc_ecc"):
        es = [t[p][m]["e"] for t in trials if accepted(t[p][m], m) and t[p][m]["e"] < 5]
        print(f"- {p} {m}: median {np.median(es):.2f} px, p90 {np.percentile(es, 90):.2f} px (n={len(es)})")
es = [t["global"]["e"] for t in trials if accepted(t["global"], "global") and t["global"]["e"] < 5]
print(f"- global: median {np.median(es):.2f} px, p90 {np.percentile(es, 90):.2f} px (n={len(es)})")

print("\n## Agreement check, case by case: which fix was right\n")
print("| prior | condition | crop | win_sift: error px / inliers | ncc_ecc: error px / ECC correlation |")
print("|---|---|---|---|---|")
for p in PRI:
    for t in trials:
        ws, ne = t[p]["win_sift"], t[p]["ncc_ecc"]
        if accepted(ws, "win_sift") and accepted(ne, "ncc_ecc") and abs(ws["e"] - ne["e"]) > 5:
            print(f"| {p} | {t['kind']} L{t['lvl']} | {t['i']} | {ws['e']:.2f} / {ws['conf']} | {ne['e']:.2f} / {ne['conf']:.2f} |")

# Accept-threshold sweep. The threshold is scored on the same frames it would be chosen on,
# so the best value here is optimistic: confirm it on held-out flights before adopting it.
KS = (6, 8, 10, 12, 15, 20, 25)


def sweep(rs):
    """'correct / wrong' fixes kept at each inlier threshold k."""
    return " | ".join(
        f"{sum(1 for r in rs if r['e'] is not None and (r['conf'] or 0) >= k and r['e'] < 5)} / "
        f"{sum(1 for r in rs if r['e'] is not None and (r['conf'] or 0) >= k and r['e'] > 20)}" for k in KS)


print("\n## Inlier-threshold sweep: correct (<5 px) / wrong (>20 px) fixes kept at each threshold k\n")
print("Scored on the same frames, so optimistic; confirm on held-out flights.\n")
print("| method | trials | " + " | ".join(f"k={k}" for k in KS) + " |")
print("|---|---|" + "---|" * len(KS))
print(f"| global, all | {len(trials)} | " + sweep([t["global"] for t in trials]) + " |")
print(f"| global, hard | {len(hsel)} | " + sweep([t["global"] for t in hsel]) + " |")
ws_all = [t[p]["win_sift"] for t in trials for p in PRI]
print(f"| win_sift, all priors pooled | {len(ws_all)} | " + sweep(ws_all) + " |")
