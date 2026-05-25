"""
phi_cmb_scan.py
QCore Labs — Richard L. Jackson
May 2026

Three-layer phi-cascade CMB detection pipeline.

Layer 1 — Copilot's Master Pipeline (skeleton)
    Stage 1: geometric filtering
    Stage 2: feature detection with mode switching
    Unified reporting, deterministic flow

Layer 2 — Gemini's evaluate_spot() (filter logic)
    Clean, modular, extensible
    Per-order adaptive b_cut
    Unified two-stage status output

Layer 3 — Claude's f_clip (bias metric)
    Circular segment area formula (geometrically exact)
    Embedded inside Gemini's evaluate()
    Required for soft-clip classification and sky-weighting

PRE-REGISTERED CASCADE (Penrose_phi_CMB_cutoff_v3):
    theta_c = arcsin(1/sqrt(phi)) = 51.827 deg
    m=1: 32.031 deg   m=2: 19.796 deg   m=3: 12.235 deg
    m=4:  7.562 deg   m=5:  4.673 deg   m=6:  2.888 deg

USAGE:
    python3 phi_cmb_scan.py --data spots.csv
    python3 phi_cmb_scan.py --simulate
    CSV columns: l_deg, b_deg, diameter_deg
"""

import numpy as np
from scipy.stats import binomtest, norm
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

PHI = (1 + np.sqrt(5)) / 2

# ── Cascade definitions ────────────────────────────────────────────────────────
def theta_c_deg():
    return np.degrees(np.arcsin(np.sqrt(1 / PHI)))

def cascade_deg():
    tc = theta_c_deg()
    return {m: tc * PHI**(-m) for m in range(1, 7)}

def print_predictions():
    print("=" * 62)
    print("PHI-CASCADE PRE-REGISTERED ANGULAR PREDICTIONS")
    print("QCore Labs | Jackson (2026)")
    print("=" * 62)
    print(f"phi           = {PHI:.10f}")
    print(f"theta_c       = {theta_c_deg():.6f} deg")
    print(f"sin^2(theta_c) = 1/phi = {1/PHI:.10f}")
    roles = {1:"first sub-critical", 2:"second-order", 3:"third-order",
             4:"PRIMARY TEST", 5:"Penrose cutoff", 6:"CMB-S4 range"}
    print(f"\n  {'m':>3}  {'Degrees':>9}  {'Radians':>10}  Role")
    print(f"  " + "-" * 52)
    for m, d in cascade_deg().items():
        print(f"  {m:>3}  {d:>9.4f}  {np.radians(d):>10.6f}  {roles[m]}")
    print("=" * 62)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Claude's f_clip (circular segment, geometrically exact)
# Embedded here so Gemini's evaluate() can call it
# ══════════════════════════════════════════════════════════════════════════════

def clipping_fraction(b_center, diameter, b_cut):
    """
    Area fraction of spot disk occluded by galactic mask boundary.
    Uses circular segment formula — geometrically exact.
    Returns (f_clip, d_bias_fraction).
    """
    r = diameter / 2.0
    dist = abs(b_center) - b_cut
    if dist >= r:  return 0.0, 0.0
    if dist <= -r: return 1.0, 1.0
    h = r - dist
    cos_arg = np.clip((r - h) / r, -1, 1)
    seg = r**2 * np.arccos(cos_arg) - (r - h) * np.sqrt(max(0, 2*r*h - h**2))
    f = float(seg / (np.pi * r**2))
    return f, float(f * 0.5)

def sky_weight(b_cut, radius):
    """Sky availability weight: 1 / (1 - sin(b_cut + radius))."""
    avail = 1.0 - np.sin(np.radians(b_cut + radius))
    return float(1.0 / avail) if avail > 1e-6 else np.inf

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Gemini's evaluate_spot() with Claude's f_clip embedded
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_spot(b_center, diameter, m_order, b_base=30.0, alpha=1.0,
                  f_clip_threshold=0.05):
    """
    Gemini's unified two-stage spot evaluation.
    Claude's clipping_fraction() provides the bias metric.

    Stage 1 hard_b_cut = b_base + diameter/2  (alpha=1 enforced)
    Stage 2 adaptive   = b_base + alpha*radius (tunable)

    Returns
    -------
    stage1      : 'ACCEPTED' or 'REJECTED'
    stage2      : 'SAFE', 'SOFT_CLIP', 'CLIPPED', 'MASKED'
    f_clip      : area fraction occluded (Claude's circular segment)
    d_bias      : estimated diameter bias fraction
    sky_wt      : sky availability correction factor
    b_cut       : adaptive b_cut applied
    """
    radius = diameter / 2.0
    center_dist = abs(b_center)

    # Stage 1: strict containment (alpha=1 hardcoded by design)
    hard_cut = b_base + radius
    stage1 = ('ACCEPTED' if center_dist > hard_cut and
              center_dist - hard_cut >= radius else 'REJECTED')

    # Stage 2: adaptive cut
    b_cut = b_base + alpha * radius
    dist_to_edge = center_dist - b_cut

    if center_dist <= b_cut:
        return stage1, 'MASKED', 1.0, 1.0, np.nan, b_cut

    # Layer 3: Claude's area-fraction f_clip
    f_clip, d_bias = clipping_fraction(b_center, diameter, b_cut)

    wt = sky_weight(b_cut, radius)

    if f_clip == 0.0:
        stage2 = 'SAFE'
    elif f_clip <= f_clip_threshold:
        stage2 = 'SOFT_CLIP'
    else:
        stage2 = 'CLIPPED'
        wt = np.nan

    return stage1, stage2, round(f_clip, 4), round(d_bias, 4), wt, b_cut

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Copilot's Master Pipeline (skeleton, exact structure)
# ══════════════════════════════════════════════════════════════════════════════

def geometric_filter(l_arr, b_arr, d_arr, b_base=30.0, alpha=1.0,
                     f_clip_threshold=0.05, include_soft=False):
    """Stage 1: geometric filtering. Returns safe diameters + full status."""
    cascade = cascade_deg()

    def nearest_m(d):
        return min(cascade.keys(), key=lambda m: abs(cascade[m] - d))

    evals = [evaluate_spot(b, d, nearest_m(d), b_base, alpha, f_clip_threshold)
             for b, d in zip(b_arr, d_arr)]

    stage1   = np.array([e[0] for e in evals])
    stage2   = np.array([e[1] for e in evals])
    f_clips  = np.array([e[2] for e in evals])
    d_biases = np.array([e[3] for e in evals])
    sky_wts  = np.array([e[4] if not (isinstance(e[4], float) and np.isnan(e[4]))
                         else 0.0 for e in evals])

    s1_safe = stage1 == 'ACCEPTED'
    s2_safe = (stage2 == 'SAFE') | (include_soft & (stage2 == 'SOFT_CLIP'))

    statuses = stage2  # use Stage 2 for reporting granularity

    print(f"\nGeometric Filter (b_base={b_base}, alpha={alpha})")
    print(f"  Input     : {len(d_arr):>5}")
    print(f"  Stage 1 ACCEPTED : {np.sum(s1_safe):>4}  (hard containment)")
    print(f"  SAFE      : {np.sum(stage2=='SAFE'):>4}  (f_clip = 0)")
    print(f"  SOFT_CLIP : {np.sum(stage2=='SOFT_CLIP'):>4}  (0 < f_clip <= {f_clip_threshold})")
    print(f"  CLIPPED   : {np.sum(stage2=='CLIPPED'):>4}  (f_clip > {f_clip_threshold})")
    print(f"  MASKED    : {np.sum(stage2=='MASKED'):>4}")

    return d_arr[s1_safe], d_arr[s2_safe], statuses, f_clips, sky_wts

def binomial_detection(diameters, cascade, window_frac=0.10):
    """
    Copilot's small-N detector.
    Correct for N < 100. Asks: is clustering more than uniform chance?
    """
    results = {}
    N = len(diameters)
    D_RANGE = 34.0  # 1 to 35 deg

    for m, pred in cascade.items():
        wlo = pred * (1 - window_frac)
        whi = pred * (1 + window_frac)
        observed = int(np.sum((diameters >= wlo) & (diameters <= whi)))
        p_window = (whi - wlo) / D_RANGE
        expected = N * p_window
        pval = binomtest(observed, N, p_window, alternative='greater').pvalue
        equiv_sigma = float(-norm.ppf(pval)) if 0 < pval < 1 else (8.0 if pval == 0 else 0.0)

        results[m] = {
            "pred":        pred,
            "observed":    observed,
            "expected":    round(expected, 3),
            "p_value":     round(pval, 6),
            "equiv_sigma": round(equiv_sigma, 3),
            "significant": pval < 0.01,
            "status":      ('DETECTED'     if pval < 0.01 else
                            'MARGINAL'     if pval < 0.05 else
                            'NOT DETECTED'),
            "method":      "binomial"
        }
    return results

def histogram_sigma_detection(diameters, cascade, bin_width=0.4):
    """
    Copilot's large-N detector.
    Correct for N >= 100. Histogram + Gaussian baseline.
    """
    bins = np.arange(1.0, 35.5, bin_width)
    counts, edges = np.histogram(diameters, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    baseline = ndimage.gaussian_filter1d(counts.astype(float), sigma=3.0)

    results = {}
    for m, pred in cascade.items():
        wlo, whi = pred * 0.90, pred * 1.10
        inw = (centers >= wlo) & (centers <= whi)
        if not np.any(inw):
            results[m] = {"pred": pred, "peak": 0, "base": 0,
                          "excess": 0, "status": "NO BINS", "method": "histogram"}
            continue
        peak   = float(np.max(counts[inw]))
        base   = float(np.mean(baseline[inw]))
        sigma  = np.sqrt(max(base, 1.0))
        excess = (peak - base) / sigma
        results[m] = {
            "pred":        pred,
            "peak":        peak,
            "base":        base,
            "equiv_sigma": round(excess, 3),
            "p_value":     None,
            "significant": excess > 2.0,
            "status":      ('DETECTED'     if excess > 2.0 else
                            'MARGINAL'     if excess > 1.0 else
                            'NOT DETECTED'),
            "method":      "histogram"
        }
    return results

def run_phi_cascade_pipeline(csv_data, b_base=30.0, alpha=1.0,
                              f_clip_threshold=0.05, include_soft=False,
                              small_n_threshold=100, bin_width=0.4):
    """
    Copilot's master pipeline — exact structure preserved.
    Stage 1: geometric filtering
    Stage 2: feature detection with automatic mode switching
    """
    l_arr = np.array([x[0] for x in csv_data])
    b_arr = np.array([x[1] for x in csv_data])
    d_arr = np.array([x[2] for x in csv_data])

    # Stage 1: geometric filter
    d_s1, d_s2, statuses, f_clips, sky_wts = geometric_filter(
        l_arr, b_arr, d_arr, b_base, alpha, f_clip_threshold, include_soft)

    cascade = cascade_deg()
    N = len(d_s1)

    # Stage 2: mode-switched detection
    if N < small_n_threshold:
        detection_s1 = binomial_detection(d_s1, cascade)
        detection_s2 = binomial_detection(d_s2, cascade)
        mode = "BINOMIAL_SMALL_N"
    else:
        detection_s1 = histogram_sigma_detection(d_s1, cascade, bin_width)
        detection_s2 = histogram_sigma_detection(d_s2, cascade, bin_width)
        mode = "HISTOGRAM_LARGE_N"

    return {
        "N_input":     len(csv_data),
        "N_safe_s1":   len(d_s1),
        "N_safe_s2":   len(d_s2),
        "statuses":    statuses,
        "f_clips":     f_clips,
        "sky_weights": sky_wts,
        "mode":        mode,
        "detection_s1": detection_s1,
        "detection_s2": detection_s2,
        "d_s1":        d_s1,
        "d_s2":        d_s2,
    }

# ── Reporting ──────────────────────────────────────────────────────────────────
def print_results(result):
    mode = result["mode"]
    print(f"\n{'='*62}")
    print(f"FEATURE DETECTION  [{mode}]")
    print(f"{'='*62}")

    for label, det in [("Stage 1 — Hard Geometry", result["detection_s1"]),
                       ("Stage 2 — Adaptive",      result["detection_s2"])]:
        n = result["N_safe_s1"] if "Stage 1" in label else result["N_safe_s2"]
        print(f"\n  {label} ({n} spots)")
        if mode == "BINOMIAL_SMALL_N":
            print(f"  {'m':>3}  {'Pred°':>7}  {'N_in':>5}  {'Expect':>7}  "
                  f"{'p-value':>9}  {'~σ':>6}  Status")
            print(f"  " + "-" * 58)
            for m, r in det.items():
                mk = " <<<" if r["status"] == "DETECTED" else ""
                print(f"  {m:>3}  {r['pred']:>7.3f}  {r['observed']:>5}  "
                      f"{r['expected']:>7.3f}  {r['p_value']:>9.6f}  "
                      f"{r['equiv_sigma']:>6.2f}  {r['status']}{mk}")
        else:
            print(f"  {'m':>3}  {'Pred°':>7}  {'Peak':>6}  {'Base':>7}  "
                  f"{'σ':>7}  Status")
            print(f"  " + "-" * 50)
            for m, r in det.items():
                mk = " <<<" if r["status"] == "DETECTED" else ""
                print(f"  {m:>3}  {r['pred']:>7.3f}  {r.get('peak',0):>6.1f}  "
                      f"{r.get('base',0):>7.2f}  {r['equiv_sigma']:>7.2f}  "
                      f"{r['status']}{mk}")

    print(f"\n{'='*62}")
    print("CROSS-VALIDATION")
    print(f"{'='*62}")
    print(f"  {'m':>3}  {'Pred°':>7}  {'Stage1':>12}  {'Stage2':>12}  Verdict")
    print(f"  " + "-" * 58)
    cascade = cascade_deg()
    for m in cascade:
        s1 = result["detection_s1"].get(m, {}).get("status", "---")
        s2 = result["detection_s2"].get(m, {}).get("status", "---")
        verdict = ("STRONG CONFIRMATION" if s1 == "DETECTED" and s2 == "DETECTED" else
                   "Stage1 only"         if s1 == "DETECTED" else
                   "Stage2 only"         if s2 == "DETECTED" else
                   "marginal"            if "MARGINAL" in (s1, s2) else
                   "not detected")
        print(f"  {m:>3}  {cascade[m]:>7.3f}  {s1:>12}  {s2:>12}  {verdict}")

    m4_s1 = result["detection_s1"].get(4, {}).get("status", "---")
    m4_s2 = result["detection_s2"].get(4, {}).get("status", "---")
    print(f"\n  PRIMARY TEST (m=4 at 7.562 deg):")
    if m4_s1 == "DETECTED" and m4_s2 == "DETECTED":
        print(f"  STRONG POSITIVE — confirmed in both stages")
    elif m4_s1 == "DETECTED":
        print(f"  POSITIVE in Stage 1 only")
    elif m4_s2 == "DETECTED":
        print(f"  POSITIVE in Stage 2 only — check bias")
    else:
        print(f"  NOT DETECTED")

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_results(result, output_path, title):
    fig = plt.figure(figsize=(16, 7), facecolor='#0a0e1a')
    gs  = fig.add_gridspec(1, 3, width_ratios=[3, 3, 1.2], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    for a in (ax1, ax2, ax3): a.set_facecolor('#0a0e1a')

    cascade = cascade_deg()
    colors = {1:'#ff6b35',2:'#ffa500',3:'#ffdd00',
              4:'#00ff9d',5:'#00cfff',6:'#bf7fff'}

    def draw_panel(ax, diameters, det, label):
        bins = np.arange(1.0, 35.5, 0.4)
        counts, edges = np.histogram(diameters, bins=bins)
        centers = 0.5*(edges[:-1]+edges[1:])
        bw = centers[1]-centers[0] if len(centers)>1 else 0.4
        ax.bar(centers, counts, width=bw, color='#1e3a5f',
               edgecolor='#2a5a8f', lw=0.5, alpha=0.9)
        baseline = ndimage.gaussian_filter1d(counts.astype(float), sigma=3.0)
        ax.plot(centers, baseline, color='#4a9eff', lw=1.2, ls='--', alpha=0.7)
        for m, pred in cascade.items():
            c = colors[m]
            if 1.0 <= pred <= 35.0:
                st = det.get(m, {}).get("status","---")
                lw = 2.5 if m == 4 else 1.5
                ls = '-' if st == 'DETECTED' else '--'
                ax.axvline(pred, color=c, lw=lw, ls=ls, alpha=0.9,
                           label=f'm={m} {pred:.1f}°')
                ax.axvspan(pred*0.90, pred*1.10, alpha=0.07, color=c)
        ax.set_title(label, color='white', fontsize=10, pad=8)
        ax.set_xlabel('Angular Diameter (deg)', color='white', fontsize=9)
        ax.set_ylabel('Count', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a4a6a')
        ax.legend(fontsize=6.5, facecolor='#0f1929', edgecolor='#2a4a6a',
                  labelcolor='white', ncol=2, loc='upper right')

    draw_panel(ax1, result["d_s1"], result["detection_s1"],
               f"Stage 1 — Hard Geometry ({result['N_safe_s1']} spots)")
    draw_panel(ax2, result["d_s2"], result["detection_s2"],
               f"Stage 2 — Adaptive ({result['N_safe_s2']} spots)")

    # Cross-validation sigma bar chart
    m_vals = list(range(1, 7))
    s1_sig = [result["detection_s1"].get(m,{}).get("equiv_sigma",0) or 0 for m in m_vals]
    s2_sig = [result["detection_s2"].get(m,{}).get("equiv_sigma",0) or 0 for m in m_vals]
    x = np.arange(1, 7)
    ax3.bar(x-0.18, s1_sig, 0.32, color='#4a9eff', alpha=0.85, label='Stage 1')
    ax3.bar(x+0.18, s2_sig, 0.32, color='#00ff9d', alpha=0.85, label='Stage 2')
    ax3.axhline(2.0, color='#ffdd00', lw=1.5, ls='--', alpha=0.8, label='p=0.01')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'm={m}' for m in m_vals], color='white', fontsize=7)
    ax3.set_ylabel('~σ', color='white', fontsize=9)
    ax3.set_title('Cross-Val', color='white', fontsize=9, pad=6)
    ax3.tick_params(colors='white', labelsize=7)
    for sp in ax3.spines.values(): sp.set_edgecolor('#2a4a6a')
    ax3.legend(fontsize=7, facecolor='#0f1929', edgecolor='#2a4a6a', labelcolor='white')

    mode_label = result["mode"].replace("_"," ")
    fig.suptitle(f"{title}  [{mode_label}]", color='white', fontsize=12, y=0.99)
    plt.savefig(output_path, dpi=150, facecolor='#0a0e1a', bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()

# ── Data I/O ───────────────────────────────────────────────────────────────────
def load_csv(filepath):
    import csv
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        l_col = next((h for h in headers if 'l' in h.lower()), None)
        b_col = next((h for h in headers if 'b' in h.lower()), None)
        d_col = next((h for h in headers if 'diam' in h.lower() or 'angular' in h.lower()), None)
        if not (l_col and b_col and d_col):
            raise ValueError(f"Need l_deg, b_deg, diameter_deg. Found: {headers}")
        for row in reader:
            rows.append((float(row[l_col]), float(row[b_col]), float(row[d_col])))
    return rows

def generate_demo(n=800, inject=True, seed=42):
    rng = np.random.default_rng(seed)
    cascade = cascade_deg()
    data = []
    while len(data) < n:
        b = np.degrees(np.arcsin(rng.uniform(0.25,1.0))) * rng.choice([-1,1])
        l = rng.uniform(0, 360)
        d = rng.exponential(4.5) + 1.0
        if 1.0 < d < 35.0:
            data.append((l, b, d))
    if inject:
        for m, deg in cascade.items():
            if deg < 1.5: continue
            for _ in range(max(4, int(15/m))):
                b_min = 30.0 + deg/2.0 + 2.0
                b = rng.uniform(b_min, 85.0) * rng.choice([-1,1])
                data.append((rng.uniform(0,360), b,
                             rng.normal(deg, deg*0.07)))
    return [(l,b,d) for l,b,d in data if 1.0 < d < 35.0]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='phi-CMB Cascade Pipeline — QCore Labs')
    parser.add_argument('--data',      type=str, default=None)
    parser.add_argument('--simulate',  action='store_true')
    parser.add_argument('--no-inject', dest='inject', action='store_false', default=True)
    parser.add_argument('--b-base',    type=float, default=30.0)
    parser.add_argument('--alpha',     type=float, default=1.0)
    parser.add_argument('--output',    type=str,
        default='/mnt/user-data/outputs/phi_cmb_cascade_scan.png')
    args = parser.parse_args()

    print_predictions()

    if args.data:
        csv_data = load_csv(args.data)
        title = f"phi-Cascade CMB — Real Data ({len(csv_data)} spots)"
    else:
        csv_data = generate_demo(inject=args.inject)
        mode = "injected" if args.inject else "background"
        title = f"phi-Cascade CMB — Demo ({mode})"

    print(f"\nTotal input: {len(csv_data)} spots")

    result = run_phi_cascade_pipeline(
        csv_data, b_base=args.b_base, alpha=args.alpha)

    print_results(result)
    plot_results(result, args.output, title)

if __name__ == '__main__':
    main()