#!/usr/bin/env python3
"""
Monte Carlo evaluation of pod placement and redundancy in Kubernetes.

Produces the results used in the paper: architectures (single replica, same
node, multiple nodes), scheduler sensitivity, analytical cross-check, replica
count per availability class, and the two figures.

Usage:  python3 ha_montecarlo.py [--runs 5000] [--seed 2026] [--out results] [--no-figures]
Needs:  numpy, matplotlib (scipy is optional)
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

# Parameters
T_OBS = 8760.0                      # observation window: one year, in hours
K_HW = 0.7                          # Weibull shape for hardware (Schroeder & Gibson, DSN 2006)
AFR_HW = 0.08                       # first-year hardware failure probability (Vishwanath & Nagappan, 2010)
# Weibull scale solved from 1 - exp(-(T_OBS / lambda)^k) = AFR
LAMBDA_HW = T_OBS / (-math.log(1.0 - AFR_HW)) ** (1.0 / K_HW)
MTTF_HW = LAMBDA_HW * math.gamma(1.0 + 1.0 / K_HW)
MTTF_SW = 650.0                     # software MTTF per pod, about 13 failures a year
MU_R, SIG_R = math.log(3.0), 0.7    # log-normal repair time, median 3 h
MEAN_R = math.exp(MU_R + SIG_R ** 2 / 2.0)
N_PODS, M_NODES = 4, 4
ALPHAS = [1.05, 1.10, 1.15, 1.20, 1.30]
SLA = [("Class 1 (99.9%)", 0.999), ("Class 2 (99.5%)", 0.995), ("Class 3 (99.0%)", 0.990)]
CLASS_TARGETS = [0.999, 0.995, 0.990]
DRAW_BUFFER = 150                   # failures drawn per run in the paired experiment


# Interval arithmetic on down periods: union = series, intersection = parallel
def _merge(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def union(*lists):
    allv = []
    for l in lists:
        allv.extend(l)
    return _merge(allv)


def intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def length(iv):
    return sum(e - s for s, e in iv)


# Simulation of one component and of each architecture
def component_down(rng, kind):
    """Down intervals of a hardware ('hw') or software ('sw') component over one year."""
    iv, t = [], 0.0
    while True:
        t += rng.weibull(K_HW) * LAMBDA_HW if kind == "hw" else rng.exponential(MTTF_SW)
        if t >= T_OBS:
            break
        r = rng.lognormal(MU_R, SIG_R)
        iv.append((t, min(t + r, T_OBS)))
        t += r
    return iv


def sim_single(rng):
    # hardware and software in series
    return length(union(component_down(rng, "hw"), component_down(rng, "sw")))


def sim_same(rng):
    # one node (hardware) in series with N_PODS pods in parallel
    hw = component_down(rng, "hw")
    pods = [component_down(rng, "sw") for _ in range(N_PODS)]
    allpods = pods[0]
    for p in pods[1:]:
        allpods = intersect(allpods, p)
    return length(union(hw, allpods))


def sim_multi(rng):
    # M_NODES independent nodes in parallel
    nodes = [union(component_down(rng, "hw"), component_down(rng, "sw")) for _ in range(M_NODES)]
    d = nodes[0]
    for nd in nodes[1:]:
        d = intersect(d, nd)
    return length(d)


def sim_sched_paired(rng, alphas):
    """Software layer of one node. The same draws are reused for every alpha,
    with up times divided by alpha, so the schedulers are compared pairwise."""
    e = rng.exponential(1.0, DRAW_BUFFER) * MTTF_SW
    r = rng.lognormal(MU_R, SIG_R, DRAW_BUFFER)
    out = []
    for a in alphas:
        t = d = 0.0
        for i in range(DRAW_BUFFER):
            t += e[i] / a
            if t >= T_OBS:
                break
            end = min(t + r[i], T_OBS)
            d += end - t
            t += r[i]
        else:
            raise RuntimeError("DRAW_BUFFER too small")
        out.append(d)
    return out


def run_many(fn, n, master_seed, tag):
    streams = np.random.SeedSequence([master_seed, tag]).spawn(n)
    return np.array([fn(np.random.Generator(np.random.PCG64(s))) for s in streams])


def run_paired(n, master_seed, tag):
    streams = np.random.SeedSequence([master_seed, tag]).spawn(n)
    rows = [sim_sched_paired(np.random.Generator(np.random.PCG64(s)), [1.0] + ALPHAS) for s in streams]
    return np.array(rows)           # downtime in hours, one column per alpha (first column: alpha = 1)


# Statistics
def stats(dt):
    a = 1.0 - dt / T_OBS
    mu = a.mean()
    return dict(
        mean=100 * mu, std=100 * a.std(ddof=1),
        lo=100 * float(np.percentile(a, 2.5)), hi=100 * float(np.percentile(a, 97.5)),
        nines=(-math.log10(1.0 - mu) if mu < 1.0 else float("inf")),
        dt=float(dt.mean()), se_dt=float(dt.std(ddof=1) / math.sqrt(len(dt))),
        n_zero=int((dt == 0).sum()), n=int(len(dt)),
    )


def paired_diff(dt_a, dt_b):
    """Mean and 95 % interval of dt_b - dt_a."""
    d = dt_b - dt_a
    m = float(d.mean())
    h = 1.96 * float(d.std(ddof=1)) / math.sqrt(len(d))
    return m, m - h, m + h


# Analytical availability over the one-year window.
# The point availability U(t) of a component that starts up at t = 0 satisfies
# U = S + (f*g)*U, where S is the survival function of the up time and f*g the
# density of an up + repair cycle. It is solved on a time grid.
try:
    from scipy.special import erf as _erf
except Exception:
    _erf = np.vectorize(math.erf)


def _lognorm_cdf(t):
    out = np.zeros_like(t)
    m = t > 0
    out[m] = 0.5 * (1.0 + _erf((np.log(t[m]) - MU_R) / (SIG_R * math.sqrt(2.0))))
    return out


def point_availability(surv, dt=0.25):
    n = int(round(T_OBS / dt))
    t = np.arange(n + 1) * dt
    S = surv(t)
    pu = -np.diff(S)                    # probability mass of the up time in each bin
    pr = np.diff(_lognorm_cdf(t))       # same for the repair time
    conv = np.convolve(pu, pr)
    out = []
    # A cycle can be assigned to bin i+j-1 (slightly too short) or i+j (slightly
    # too long); the average of the two removes the first-order grid error.
    for shift in (1, 2):
        pc = np.zeros(n + 1)
        pc[shift:] = conv[:n + 1 - shift]
        U = np.empty(n + 1)
        U[0] = 1.0
        for m in range(1, n + 1):
            U[m] = S[m] + np.dot(pc[1:m + 1], U[m - 1::-1])
        out.append(U)
    return 0.5 * (out[0] + out[1])


def _mean_over_horizon(x, dt):
    return float(dt * (0.5 * x[0] + x[1:-1].sum() + 0.5 * x[-1]) / T_OBS)


def analytical(dt=0.25):
    Uhw = point_availability(lambda t: np.exp(-(t / LAMBDA_HW) ** K_HW), dt)
    Usw = point_availability(lambda t: np.exp(-t / MTTF_SW), dt)
    down_single = 1.0 - Uhw * Usw
    down_same = 1.0 - Uhw * (1.0 - (1.0 - Usw) ** N_PODS)
    down_multi = (1.0 - Uhw * Usw) ** M_NODES
    res = {}
    for name, d in (("single", down_single), ("same", down_same), ("multi", down_multi)):
        u = _mean_over_horizon(d, dt)
        res[name] = dict(unavail=u, avail=100 * (1 - u), dt=u * T_OBS,
                         nines=-math.log10(u) if u > 0 else float("inf"))
    res["hw_dt"] = _mean_over_horizon(1.0 - Uhw, dt) * T_OBS
    res["sw_dt"] = _mean_over_horizon(1.0 - Usw, dt) * T_OBS
    # steady-state values, for comparison
    a_hw = MTTF_HW / (MTTF_HW + MEAN_R)
    a_sw = MTTF_SW / (MTTF_SW + MEAN_R)
    res["stationary"] = dict(
        a_hw=100 * a_hw, a_sw=100 * a_sw, a_node=100 * a_hw * a_sw,
        a_same=100 * a_hw * (1 - (1 - a_sw) ** N_PODS),
        a_multi_nines=-math.log10((1 - a_hw * a_sw) ** M_NODES),
    )
    # software layer for each alpha
    sch = {}
    for a in ALPHAS:
        U = point_availability(lambda t, a=a: np.exp(-t * a / MTTF_SW), dt)
        sch[a] = dict(avail=100 * (1 - _mean_over_horizon(1.0 - U, dt)),
                      dt=_mean_over_horizon(1.0 - U, dt) * T_OBS)
    sch[1.0] = dict(avail=100 * (1 - _mean_over_horizon(1.0 - Usw, dt)),
                    dt=_mean_over_horizon(1.0 - Usw, dt) * T_OBS)
    res["sched"] = sch
    return res


# Replica count and node floor per availability class
def class_table(a_ref):
    rows = []
    for i, ac in enumerate(CLASS_TARGETS, start=1):
        r = max(1, math.ceil(math.log(1 - ac) / math.log(1 - a_ref)))
        floor = 1.0 - (1.0 - ac) ** (1.0 / r)
        svc = 1.0 - (1.0 - a_ref) ** r
        rows.append(dict(cls=i, target=100 * ac, replicas=r, floor=100 * floor, service=100 * svc))
    return rows


# Figures
def make_figures(out, dt_mm, dt_by_alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    c_mm, c_def = "#2a78d6", "#eb6834"
    ink, ink2, axis = "#0b0b0b", "#52514e", "#c9c8c2"
    plt.rcParams.update({
        "font.size": 8, "text.color": ink, "axes.labelcolor": ink2,
        "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": axis,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Figure 1: availability histograms, one panel per alpha
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(13, 2.9), sharey=True)
    a_mm = 100 * (1 - dt_mm / T_OBS)
    lo = min(100 * (1 - dt_by_alpha[ALPHAS[-1]] / T_OBS).min(), a_mm.min())
    bins = np.linspace(lo - 0.05, 100.0, 60)
    for ax, al in zip(axes, ALPHAS):
        a_df = 100 * (1 - dt_by_alpha[al] / T_OBS)
        ax.hist(a_mm, bins=bins, color=c_mm, alpha=0.55, linewidth=0)
        ax.hist(a_df, bins=bins, color=c_def, alpha=0.55, linewidth=0)
        ax.axvline(a_mm.mean(), color=c_mm, ls="--", lw=1.4)
        ax.axvline(a_df.mean(), color=c_def, ls="--", lw=1.4)
        ax.set_title(rf"$\alpha$ = {al:.2f}", fontsize=9, color=ink)
        ax.set_xlabel("Availability (%)")
        ax.text(0.03, 0.97, f"Multi-Metric  {dt_mm.mean():.1f} h/yr\nDefault          {dt_by_alpha[al].mean():.1f} h/yr",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.8, color=ink2)
    axes[0].set_ylabel("Runs")
    fig.legend(handles=[Patch(color=c_mm, alpha=0.7, label="Multi-Metric ($\\alpha$ = 1)"),
                        Patch(color=c_def, alpha=0.7, label="Default (CPU+Mem)"),
                        Line2D([0], [0], color=ink2, ls="--", lw=1.3, label="mean")],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "fig_scheduler_hists.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: share of runs meeting each class (bars: default scheduler, dashed line: multi-metric)
    fig, axes = plt.subplots(1, len(SLA), figsize=(7.2, 2.6), sharey=True)
    x = np.arange(len(ALPHAS))
    for ax, (lab, thr) in zip(axes, SLA):
        p_def = [100 * np.mean((1 - dt_by_alpha[a] / T_OBS) >= thr) for a in ALPHAS]
        p_mm = 100 * np.mean((1 - dt_mm / T_OBS) >= thr)
        ax.bar(x, p_def, 0.6, color=c_def)
        ax.axhline(p_mm, color=c_mm, ls="--", lw=1.6)
        ax.set_title(lab, fontsize=9, color=ink)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], fontsize=7)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylim(0, 108)
        n_mm = int(np.sum((1 - dt_mm / T_OBS) >= thr))
        n_df = int(sum(np.sum((1 - dt_by_alpha[a] / T_OBS) >= thr) for a in ALPHAS))
        if max(p_def) < 1.0:
            # almost no run reaches this class: give the counts instead of bars
            ax.text(0.5, 0.55, f"Multi-Metric: {n_mm} / {len(dt_mm)} runs\nDefault (all $\\alpha$): {n_df} / {len(dt_mm)} runs",
                    transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=ink2)
        else:
            for i in range(len(ALPHAS)):
                if p_def[i] > 8:
                    ax.text(x[i], p_def[i] - 2.0, f"{p_def[i]:.1f}", ha="center", va="top", fontsize=6.8, color=ink)
            ax.text(x[-1] + 0.45, p_mm + 1.5, f"{p_mm:.1f}", ha="right", va="bottom", fontsize=7, color=ink2)
    axes[0].set_ylabel("Runs meeting the class (%)")
    fig.legend(handles=[Patch(color=c_def, label="Default scheduler"),
                        Line2D([0], [0], color=c_mm, ls="--", lw=1.6, label="Multi-Metric ($\\alpha$ = 1)")],
               loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig_sla_compliance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n, seed = args.runs, args.seed
    t0 = time.time()

    print(f"lambda_hw = {LAMBDA_HW:,.0f} h   MTTF_hw = {MTTF_HW:,.0f} h   mean repair = {MEAN_R:.3f} h")
    print(f"P(first-year hw failure) = {1 - math.exp(-(T_OBS / LAMBDA_HW) ** K_HW):.4f}  (target {AFR_HW})")

    # architectures
    dt_single = run_many(sim_single, n, seed, 1)
    dt_same = run_many(sim_same, n, seed, 2)
    dt_multi = run_many(sim_multi, n, seed, 3)
    arch = {"single": stats(dt_single), "same": stats(dt_same), "multi": stats(dt_multi)}

    # scheduler comparison
    paired = run_paired(n, seed, 4)
    dt_mm = paired[:, 0]
    dt_by_alpha = {a: paired[:, i + 1] for i, a in enumerate(ALPHAS)}
    sched = {"mm": stats(dt_mm)}
    for a in ALPHAS:
        s = stats(dt_by_alpha[a])
        s["delta"], s["delta_lo"], s["delta_hi"] = paired_diff(dt_mm, dt_by_alpha[a])
        s["sla"] = {lab: 100 * float(np.mean((1 - dt_by_alpha[a] / T_OBS) >= thr)) for lab, thr in SLA}
        sched[a] = s
    sched["mm"]["sla"] = {lab: 100 * float(np.mean((1 - dt_mm / T_OBS) >= thr)) for lab, thr in SLA}

    ana = analytical()
    classes = class_table(ana["stationary"]["a_node"] / 100.0)

    # console report
    print("\nArchitectures (N=%d)" % n)
    for k, lab in (("single", "Single-Replica"), ("same", "Same-Node (4 pods)"), ("multi", "Multi-Node (4 nodes)")):
        s = arch[k]
        print(f"  {lab:22s} mean={s['mean']:.4f}%  std={s['std']:.4f}%  range95=[{s['lo']:.3f},{s['hi']:.3f}]  "
              f"nines={s['nines']:.3f}  DT={s['dt']:.3f}h (SE {s['se_dt']:.3f})  runs_without_downtime={s['n_zero']}")
    print("\nAnalytical, one year from a fresh start")
    for k in ("single", "same", "multi"):
        a = ana[k]
        print(f"  {k:7s} A={a['avail']:.5f}%  DT={a['dt']:.4f}h  nines={a['nines']:.3f}")
    print(f"  hw-only DT = {ana['hw_dt']:.4f} h   sw-only DT = {ana['sw_dt']:.3f} h")
    st = ana["stationary"]
    print(f"  steady state: A_hw={st['a_hw']:.4f}%  A_sw={st['a_sw']:.4f}%  A_node={st['a_node']:.4f}%  "
          f"A_same={st['a_same']:.5f}%  multi nines={st['a_multi_nines']:.3f}")
    print("\nScheduler comparison (software layer, paired)")
    s = sched["mm"]
    print(f"  MM alpha=1.00 mean={s['mean']:.3f}% nines={s['nines']:.3f} DT={s['dt']:.2f}h   "
          f"analytic DT={ana['sched'][1.0]['dt']:.2f}h")
    for a in ALPHAS:
        s = sched[a]
        print(f"  alpha={a:.2f} mean={s['mean']:.3f}% nines={s['nines']:.3f} DT={s['dt']:.2f}h  "
              f"dDT={s['delta']:+.2f} [{s['delta_lo']:.2f},{s['delta_hi']:.2f}]  ratio={s['delta'] / (a - 1):.1f}h  "
              f"analytic DT={ana['sched'][a]['dt']:.2f}h  SLA={ {k: round(v, 1) for k, v in s['sla'].items()} }")
    print("  MM SLA:", {k: round(v, 1) for k, v in sched["mm"]["sla"].items()})
    print("\nClasses (A_ref = %.4f%%)" % st["a_node"])
    for c in classes:
        print(f"  class {c['cls']}: target={c['target']:.1f}%  r={c['replicas']}  node floor={c['floor']:.4f}%  "
              f"service with r replicas={c['service']:.4f}%")
    print(f"\nrule of three: P(outage/yr) < {3.0 / n:.1e}")

    # output files
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(runs=n, seed=seed, lambda_hw=LAMBDA_HW, mttf_hw=MTTF_HW, mean_repair=MEAN_R,
                       arch=arch, analytical=ana, scheduler={str(k): v for k, v in sched.items()},
                       classes=classes), f, indent=2, default=float)
    with open(os.path.join(args.out, "table3_architectures.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean_pct", "std_pct", "range95_lo", "range95_hi", "nines", "downtime_h", "se_downtime_h", "runs_without_downtime"])
        for k in ("single", "same", "multi"):
            s = arch[k]
            w.writerow([k, s["mean"], s["std"], s["lo"], s["hi"], s["nines"], s["dt"], s["se_dt"], s["n_zero"]])
    with open(os.path.join(args.out, "table4_scheduler.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mean_pct", "nines", "downtime_h", "delta_dt_h", "ci_lo", "ci_hi"] + [l for l, _ in SLA])
        s = sched["mm"]
        w.writerow([1.0, s["mean"], s["nines"], s["dt"], 0, 0, 0] + [s["sla"][l] for l, _ in SLA])
        for a in ALPHAS:
            s = sched[a]
            w.writerow([a, s["mean"], s["nines"], s["dt"], s["delta"], s["delta_lo"], s["delta_hi"]] + [s["sla"][l] for l, _ in SLA])
    with open(os.path.join(args.out, "table6_classes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "target_pct", "replicas", "node_floor_pct", "service_avail_pct"])
        for c in classes:
            w.writerow([c["cls"], c["target"], c["replicas"], c["floor"], c["service"]])

    if not args.no_figures:
        make_figures(args.out, dt_mm, dt_by_alpha)
    print(f"\nDone in {time.time() - t0:.0f} s -> {args.out}/")


if __name__ == "__main__":
    main()
