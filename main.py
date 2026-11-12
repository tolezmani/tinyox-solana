#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry (v2 rebuild): Monte Carlo forecaster for
the top 5 new pump.fun coins by market cap, 24 hours out.

Rebuilt from scratch in its own session; design re-derived from the
rehearsal history (see the comments at each estimate):

  * Dynamics are estimated from the history itself: per-venue drift/vol of
    10-minute log-cap changes (pool coins drift up early, curve coins drift
    down), graduation hazard for curve coins by cap band (measured 0.00
    below 100 SOL, 0.33 at 100-300 SOL), and a newcomer process.
  * Newcomers: the rehearsal shows 23-42 new coins entering the top 60
    every 10 minutes and ~9 coins reaching the top 5 within 40 minutes of
    the opening; the top 5 at every late mark are coins aged 5-48 min
    (Ansem 0 -> #1, 2940 SOL, in ~9 min; chinese hat cat 5 min old
    already #3).  So contenders spawn at ~6-8/hour, rise with tau ~0.08 h
    to a peak ~N(3.15, 0.40) log10 SOL, then decay.  Their share of each
    place is deliberately left unnamed: the scorer shares it over the
    answer chart's un-named coins, which is exactly how a coin born after
    the cut is scored.
  * Pool coins mean-revert (OU, theta 0.5/h) to their last-mark level: a
    leader's cap hovers near where it was cut instead of random-walking
    away; curve coins random-walk with a per-hour death hazard (they fade).
  * At a short horizon the current chart mostly holds (top-5 overlap 4/5
    per 10 min late in the rehearsal), so the Monte Carlo distribution is
    blended toward the house baseline's persistence pattern with a weight
    that decays as exp(-h/2) in the horizon h (hours): ~0.65 at 20 min,
    ~0 at 13 h.
  * The unnamed share per place comes from the simulated newcomer share
    and the measured per-place newcomer takeover rate (0.50 at every place
    between consecutive marks), growing with the horizon, floored 0.45,
    capped 0.70.
  * A coin is only named at a place when its probability beats the
    unnamed per-coin share at that place (adaptive threshold): naming a
    coin with a lower p than that takes it out of the unnamed share and
    hurts.  Rows are floored to 6 decimals.
  * Sizes: a Normal in log10(cap in SOL) around the median simulated
    k-th cap, sd from the IQR clamped to [0.30, 0.60], binned like the
    starter.  top5 is ordered by total top-5 probability across reads.

Stdlib only, one fixed seed, fully deterministic.  Run locally:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""
import json
import math
import os
import random
from collections import defaultdict

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
PATHS = 2000
MAX_COINS = 320
CHART_N = 100

# --- newcomer process (measured from the rehearsal; see header) ---
FRESH_LOG = 1.45          # a fresh coin's log10 cap in SOL (~28 SOL)
NEW_PEAK_MU = 3.15        # log10 SOL; measured median peak of coins ever
NEW_PEAK_SD = 0.40        #   >300 SOL is 3.28, upper tail reaches 3.56
RISE_TAU_H = 0.08         # front-loaded rise: 1-exp(-age/tau)
NEW_DECAY_H = 0.05        # log-cap decay per hour after the peak
NEW_DEATH_H = 0.15        # per-hour death hazard after the peak

OU_THETA_H = 0.5          # pool / migrating mean reversion per hour
DEATH_H = {"pool": 0.06, "curve": 0.12, "migrating": 0.10}
SHOCK_SD_H = 0.05         # common market shock, per hour


def parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_input():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("ok") and isinstance(d.get("rows"), list):
                marks.append(d)
    marks.sort(key=lambda m: m["at"])
    return task, marks


def logcap(row):
    return math.log10(float(row["market_cap_lamports"]) / 1e9)


def estimate_dynamics(marks):
    """Per-venue 10-min log-change stats, graduation hazard by cap band,
    and the newcomer spawn rate (distinct coins reaching the top 10, per
    hour of history, clamped to a sane range)."""
    series = defaultdict(list)
    for i, m in enumerate(marks):
        for r in m["rows"]:
            series[r["mint"]].append((i, logcap(r), r["venue"], r["complete"]))
    moves = defaultdict(list)
    grad = defaultdict(lambda: [0, 0])
    for pts in series.values():
        pts.sort()
        for (a, la, va, _), (b, lb, _vb, cb) in zip(pts, pts[1:]):
            if b != a + 1:
                continue
            moves[va].append(lb - la)
            if va == "curve":
                cap = 10 ** la
                band = "lo" if cap < 100 else "mid" if cap < 300 else "hi"
                grad[band][0] += 1
                if cb:
                    grad[band][1] += 1
    stats = {}
    for venue in ("pool", "curve", "migrating"):
        xs = moves.get(venue, [])
        if xs:
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / len(xs)
            stats[venue] = (mu, math.sqrt(var) if var > 0 else 0.12)
        else:
            stats[venue] = (0.0, 0.12)
    # Pool drift in a short rehearsal is winner-biased (only coins that
    # graduated and kept rising are observed as pool); shrink it hard.
    stats["pool"] = (max(-0.01, min(0.02, stats["pool"][0] * 0.25)),
                     stats["pool"][1])
    stats["curve"] = (max(-0.03, min(0.0, stats["curve"][0])),
                      stats["curve"][1])
    hazard = {}
    for band, fallback in (("lo", 0.0), ("mid", 0.28), ("hi", 0.5)):
        n, g = grad.get(band, [0, 0])
        hazard[band] = (g / n) if n >= 5 else fallback
    # Newcomer spawn rate: distinct coins that ever reached the top 10,
    # per hour of history.  The opening flood inflates it; clamp it.
    top10 = set()
    for m in marks:
        for r in m["rows"][:10]:
            top10.add(r["mint"])
    span_h = max(1e-6, (parse_ts(marks[-1]["at"]) - parse_ts(marks[0]["at"])) / 3600.0)
    lam = max(4.0, min(8.0, len(top10) / span_h))
    return stats, hazard, series, lam


def pick_coins(marks, series):
    """Candidate universe: the last mark's chart plus anything that ever
    reached the top 60, capped at MAX_COINS by last-mark cap."""
    last = next((m for m in reversed(marks) if m["rows"]), marks[-1])
    last_cap = {}
    last_rank = {}
    for r in last["rows"]:
        last_cap[r["mint"]] = logcap(r)
        last_rank[r["mint"]] = r["rank"]
    ever = set()
    for m in marks:
        for r in m["rows"][:60]:
            ever.add(r["mint"])
    universe = set(last_cap) | ever
    venue = {}
    for m in reversed(marks):
        for r in m["rows"]:
            if r["mint"] in universe and r["mint"] not in venue:
                venue[r["mint"]] = r["venue"]
    recent = {}
    for mint in universe:
        pts = sorted(series.get(mint, []))
        ds = [b[1] - a[1] for a, b in zip(pts, pts[1:]) if b[0] == a[0] + 1]
        recent[mint] = sum(ds[-3:]) / len(ds[-3:]) if ds else None
    ordered = sorted(universe, key=lambda mn: (-last_cap.get(mn, 0.0), mn))
    return ordered[:MAX_COINS], last_cap, recent, venue, last


def bin_index(x, edges):
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges)
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if edges[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def simulate(task, marks, stats, hazard, lam, coins, last_cap, recent, venue, last):
    """Monte Carlo forward from the last mark to each answer read.
    Returns (place tallies, newcomer tallies, per-place size samples):
    tallies[read][place] = {mint: hits}, newcomers[read][place] = hits,
    sizes[read][place] = [simulated log10 caps of the k-th coin]."""
    reads = [parse_ts(s) for s in task["reads"]]
    t0 = parse_ts(last["at"])
    horizon = reads[-1] - t0
    step = 600.0 if horizon <= 3 * 3600 else 3600.0
    hours_step = step / 3600.0
    n_steps = int(math.ceil(horizon / step))
    read_step = [max(0, int((rt - t0) // step)) for rt in reads]
    by_step = defaultdict(list)
    for ri, j in enumerate(read_step):
        by_step[j].append(ri)
    n_reads = len(reads)
    tallies = [[defaultdict(int) for _ in range(5)] for _ in range(n_reads)]
    newcomers = [[0] * 5 for _ in range(n_reads)]
    size_samples = [[[] for _ in range(5)] for _ in range(n_reads)]
    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    field = [logcap(r) for r in last_rows[9:50]] or [1.5]

    n = len(coins)
    drift = []
    vol = []
    death = []
    for mn in coins:
        v = venue.get(mn, "curve")
        mu, sd = stats.get(v, (0.0, 0.12))
        r = recent.get(mn)
        d = mu if r is None else 0.5 * r + 0.5 * mu
        drift.append(d * 6.0 * hours_step)
        vol.append(sd * math.sqrt(6.0 * hours_step))
        death.append(1.0 - (1.0 - DEATH_H.get(v, 0.18)) ** hours_step)
    start_cap = [last_cap.get(mn, FRESH_LOG) for mn in coins]
    start_venue = [venue.get(mn, "curve") for mn in coins]
    lam_step = lam * hours_step
    exp_neg_lam = math.exp(-lam_step)

    rng = random.Random(SEED)
    gauss = rng.gauss
    uni = rng.random

    def tally(caps, alive, fresh, read_idxs):
        live = []
        for i in range(n):
            if alive[i]:
                live.append((caps[i], coins[i], False))
        for fc in fresh:
            if fc[4]:
                live.append((fc[0], None, True))
        live.sort(key=lambda x: (-x[0], str(x[1])))
        if len(live) < 5:  # degenerate path: pad from the field, unnamed
            for _ in range(5 - len(live)):
                live.append((rng.choice(field), None, True))
            live.sort(key=lambda x: (-x[0], str(x[1])))
        for ri in read_idxs:
            for k in range(5):
                lc, mn, is_new = live[k]
                if is_new:
                    newcomers[ri][k] += 1
                else:
                    tallies[ri][k][mn] += 1
                size_samples[ri][k].append(lc)

    for _ in range(PATHS):
        caps = list(start_cap)
        ven = list(start_venue)
        alive = [True] * n
        fresh = []  # [cap, peak, birth_h, time_to_peak_h, alive]
        if 0 in by_step:
            tally(caps, alive, fresh, by_step[0])
        for j in range(1, n_steps + 1):
            shock = gauss(0.0, SHOCK_SD_H * math.sqrt(hours_step))
            for i in range(n):
                if not alive[i]:
                    continue
                if uni() < death[i]:
                    alive[i] = False
                    caps[i] = 0.5
                    continue
                if ven[i] in ("pool", "migrating"):
                    caps[i] += (-OU_THETA_H * (caps[i] - start_cap[i]) * hours_step
                                + drift[i] + vol[i] * gauss(0.0, 1.0) + shock)
                else:
                    caps[i] += drift[i] + vol[i] * gauss(0.0, 1.0) + shock
                if ven[i] == "curve" and caps[i] > 2.0:
                    cap = 10 ** caps[i]
                    band = "lo" if cap < 100 else "mid" if cap < 300 else "hi"
                    hz = hazard.get(band, 0.0)
                    hz_step = 1.0 - (1.0 - hz) ** (6.0 * hours_step)
                    if uni() < hz_step:
                        ven[i] = "pool"
            # Poisson arrivals of new contenders (Knuth).  A newcomer is
            # born at a random instant inside this step, so it already has
            # an age in (0, step] when the step's reads are scored: in the
            # rehearsal a coin went 0 -> #1 within ~9 minutes, i.e. inside
            # a single 10-minute step.  (v1 gave newborns age 0 at their
            # birth step, so no newcomer could ever win the first read.)
            spawned = 0
            p = 1.0
            while True:
                p *= uni()
                if p <= exp_neg_lam:
                    break
                spawned += 1
            for _ in range(spawned):
                birth_h = (j - 1 + uni()) * hours_step
                fresh.append([FRESH_LOG, gauss(NEW_PEAK_MU, NEW_PEAK_SD),
                              birth_h, uni() * 0.45 + 0.15, True])
            now_h = j * hours_step
            for fc in fresh:
                if not fc[4]:
                    continue
                age_h = now_h - fc[2]
                if age_h <= fc[3]:
                    frac = 1.0 - math.exp(-age_h / RISE_TAU_H) if age_h > 0 else 0.0
                    fc[0] = FRESH_LOG + (fc[1] - FRESH_LOG) * frac
                else:
                    fc[0] = fc[1] - NEW_DECAY_H * (age_h - fc[3]) + gauss(0.0, 0.06)
                    if uni() < NEW_DEATH_H * hours_step:
                        fc[4] = False
                        fc[0] = 0.5
            if j in by_step:
                tally(caps, alive, fresh, by_step[j])
    return tallies, newcomers, size_samples


def erfc(x):
    z = abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2 - r


def above(x, mean, sd):
    if x == -float("inf"):
        return 1.0
    if x == float("inf"):
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2)))


def main():
    task, marks = load_input()
    if not marks:
        raise SystemExit("no usable marks in history")
    stats, hazard, series, lam = estimate_dynamics(marks)
    coins, last_cap, recent, venue, last = pick_coins(marks, series)
    tallies, newcomers, size_samples = simulate(
        task, marks, stats, hazard, lam, coins, last_cap, recent, venue, last)
    n_reads = len(task["reads"])
    edges = task["size"]["edges"]
    micro = 1_000_000
    horizon_h = (parse_ts(task["reads"][-1]) - parse_ts(last["at"])) / 3600.0

    # Persistence weight: the chart mostly holds over minutes, not hours.
    persist_w = 0.75 * math.exp(-horizon_h / 2.0)
    last_chart = sorted(last["rows"], key=lambda r: r["rank"])
    # Persistence pattern: a local shuffle of the last mark's chart.
    # The house baseline bets 0.30 on #k and 0.10 on each neighbour; the
    # rehearsal shows adjacent ranks swap (GTA6 #5 -> #4 in 10 min), so
    # this pattern also puts 0.05 on the coins two ranks away.
    persist = {}
    for k in range(5):
        p = {}
        if k < len(last_chart):
            p[last_chart[k]["mint"]] = 0.22
        if k > 0:
            p[last_chart[k - 1]["mint"]] = p.get(last_chart[k - 1]["mint"], 0.0) + 0.10
        if k + 1 < len(last_chart):
            p[last_chart[k + 1]["mint"]] = p.get(last_chart[k + 1]["mint"], 0.0) + 0.10
        if k > 1:
            p[last_chart[k - 2]["mint"]] = p.get(last_chart[k - 2]["mint"], 0.0) + 0.05
        if k + 2 < len(last_chart):
            p[last_chart[k + 2]["mint"]] = p.get(last_chart[k + 2]["mint"], 0.0) + 0.05
        persist[k] = p

    def floor6(p):
        u = math.floor(p * micro)
        return u / micro if u > 0 else 0.0

    probabilities = []
    sizes = []
    top5_score = defaultdict(float)
    for ri in range(n_reads):
        for k in range(5):
            for mn, c in tallies[ri][k].items():
                top5_score[mn] += c / PATHS
    for k in range(5):
        agg = defaultdict(float)
        new_share = 0.0
        for ri in range(n_reads):
            for mn, c in tallies[ri][k].items():
                agg[mn] += (c / PATHS) / n_reads
            new_share += (newcomers[ri][k] / PATHS) / n_reads
        # Unnamed share: the simulated newcomer share, or the measured
        # per-place takeover rate (0.50) grown with the horizon, whichever
        # is larger; the practice answer had newcomers on 3 of 5 places
        # 1 hour after its cut, so the share grows toward 0.70.
        unnamed = max(0.45, min(0.70, max(new_share, 0.50 + 0.012 * horizon_h)))
        for mn, p in persist[k].items():
            agg[mn] = agg.get(mn, 0.0) * (1.0 - persist_w) + p * persist_w
        total = sum(agg.values())
        if total > 0 and total > (1.0 - unnamed):
            scale = (1.0 - unnamed) / total
            for mn in agg:
                agg[mn] *= scale
        ranked = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
        # Adaptive threshold: name a coin only while its p beats the
        # unnamed per-coin share (unnamed / coins left un-named).  A row
        # below that takes its coin out of the unnamed share and hurts.
        named_here = 0
        for mn, p in ranked:
            denom = CHART_N - named_here
            if denom <= 0:
                break
            if p <= unnamed / denom:
                break
            pf = floor6(p)
            if pf > 0:
                probabilities.append({"mint": mn, "place": k + 1, "p": pf})
                named_here += 1
        # Sizes: Normal around the median simulated k-th log-cap, sd from
        # the IQR clamped to [0.32, 0.55]; pooled over reads.
        pooled = []
        for ri in range(n_reads):
            pooled.extend(size_samples[ri][k])
        pooled.sort()
        cnt = len(pooled)
        if cnt:
            med = pooled[cnt // 2]
            q1 = pooled[cnt // 4]
            q3 = pooled[(3 * cnt) // 4]
            sd = max(0.30, min(0.60, (q3 - q1) / 1.349 if q3 > q1 else 0.35))
        else:
            med, sd = 3.3, 0.35
        for b in range(len(edges) + 1):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            pf = floor6(above(lo, med, sd) - above(hi, med, sd))
            if pf > 0:
                sizes.append({"place": k + 1, "bin": int(b), "p": pf})
    top5 = [mn for mn, _ in sorted(top5_score.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:5]]
    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
