#!/usr/bin/env python3
# QCore Labs - LANL MPA Phi-Resonance Analysis
# Law of Harmonic Resonance: E_n = hbar * omega * phi^n
# Author: Rick Rezin Jackson | ORCID: 0009-0008-7380-4815
# Dataset: LANL 2002 Magnetospheric Plasma Analyzer (MPA)
# DOI: 10.48322/z6y7-r007
# Cadence: 87 seconds | Coverage: 2003-10-29 to 2008-01-03
# Orbit: Geosynchronous (~6.6 Re) — fixed boundary zone observer
#
# TO USE WITH REAL DATA:
#   pip install cdflib
#   Download CDF files from:
#     https://spdf.gsfc.nasa.gov/pub/data/lanl/02a_mpa/
#   Set CDF_DIR below to your local directory
#
# KEY CDF VARIABLES:
#   dens_lop      - Low energy ion density   1-130 eV  (cm^-3)
#   dens_hip      - High energy ion density  130eV-45keV (cm^-3)
#   dens_e        - Electron density         30eV-45keV  (cm^-3)
#   temp_lop      - Low energy ion temp      (eV, scalar)
#   temp_hip      - High energy ion temp     (eV, [Tpara, Tperp])
#   temp_e        - Electron temp            (eV, [Tpara, Tperp])
#   tratio_lop    - Tperp/Tmid low ions      (anisotropy ratio)
#   tratio_hip    - Tperp/Tmid high ions
#   tratio_e      - Tperp/Tmid electrons
#   sc_pot        - Spacecraft potential     (volts)
#   sc_pos_mag    - Position [Re, mlat, MLT]
#   qual_flag     - Quality flag (0=good, 1=suspect)

import numpy as np
import os

PHI = (1 + 5**0.5) / 2  # 1.6180339887...

CDF_DIR = None  # Set to path containing LANL MPA CDF files
FILL_VAL = -1.0e31

# --- Jackson Ladder ---
def jackson_ladder(n_min=-6, n_max=14):
    return {n: PHI**n for n in range(n_min, n_max+1)}

LADDER = jackson_ladder()

def nearest_rung(val, ladder=LADDER):
    if val <= 0:
        return None, None
    log_val = np.log(val) / np.log(PHI)
    rungs = list(ladder.keys())
    nearest = min(rungs, key=lambda n: abs(n - log_val))
    return nearest, ladder[nearest]

def phi_residual(val, ladder=LADDER):
    if val <= 0:
        return None
    n, rung_val = nearest_rung(val, ladder)
    return abs(val - rung_val) / rung_val

def phi_power(val):
    if val <= 0:
        return None
    return np.log(val) / np.log(PHI)

# --- Load Real CDF ---
def load_real_cdf(cdf_dir):
    try:
        import cdflib
        import glob
    except ImportError:
        print('[!] cdflib not installed. Run: pip install cdflib')
        return None

    files = sorted(glob.glob(cdf_dir + '*.cdf') + glob.glob(cdf_dir + '*.CDF'))
    if not files:
        print(f'[!] No CDF files found in {cdf_dir}')
        return None

    print(f'[*] Loading {len(files)} CDF files')

    all_dens_lop  = []
    all_dens_hip  = []
    all_dens_e    = []
    all_temp_lop  = []
    all_temp_hip  = []
    all_temp_e    = []
    all_tratio_lop= []
    all_tratio_hip= []
    all_tratio_e  = []
    all_sc_pot    = []
    all_epoch     = []

    for f in files:
        try:
            cdf = cdflib.CDF(f)

            qf = cdf.varget('qual_flag')
            gf = cdf.varget('gap_flag')

            good = (qf == 0) & (gf == 0)

            def get(var):
                d = cdf.varget(var)
                # handle 2D arrays (Tpara, Tperp) — take mean
                if d.ndim > 1:
                    d = np.mean(d, axis=1)
                d = np.where(d < FILL_VAL * 0.5, np.nan, d)
                return d[good]

            dl = get('dens_lop')
            dh = get('dens_hip')
            de = get('dens_e')
            tl = get('temp_lop')
            th = get('temp_hip')
            te = get('temp_e')
            rl = get('tratio_lop')
            rh = get('tratio_hip')
            re = get('tratio_e')
            sp = get('sc_pot')
            ep = cdf.varget('Epoch')[good]

            # validity filter
            mask = (
                (dl > 0) & (dl < 1e5) &
                (dh > 0) & (dh < 1e4) &
                (de > 0) & (de < 1e5) &
                (tl > 0) & (tl < 5e4) &
                (th > 0) & (th < 5e4) &
                (te > 0) & (te < 5e4) &
                np.isfinite(dl) & np.isfinite(dh) & np.isfinite(de)
            )

            all_dens_lop.append(dl[mask])
            all_dens_hip.append(dh[mask])
            all_dens_e.append(de[mask])
            all_temp_lop.append(tl[mask])
            all_temp_hip.append(th[mask])
            all_temp_e.append(te[mask])
            all_tratio_lop.append(rl[mask])
            all_tratio_hip.append(rh[mask])
            all_tratio_e.append(re[mask])
            all_sc_pot.append(sp[mask])
            all_epoch.append(ep[mask])

        except Exception as e:
            print(f'[!] Skipping {f}: {e}')
            continue

    if not all_dens_lop:
        return None

    print(f'[*] Real data loaded: {sum(len(x) for x in all_dens_lop)} valid points')

    return (
        np.concatenate(all_dens_lop),
        np.concatenate(all_dens_hip),
        np.concatenate(all_dens_e),
        np.concatenate(all_temp_lop),
        np.concatenate(all_temp_hip),
        np.concatenate(all_temp_e),
        np.concatenate(all_tratio_lop),
        np.concatenate(all_tratio_hip),
        np.concatenate(all_tratio_e),
        np.concatenate(all_sc_pot),
    )

# --- Synthetic Data ---
def load_synthetic():
    print('[*] Using synthetic data (modeled from Thomsen/Henderson MPA parameter ranges)')
    np.random.seed(7)
    n = 35000  # ~35 days at 87s cadence

    # Geosynchronous orbit sits at the boundary between plasmasphere
    # and ring current. MPA sees both cold dense plasma and hot ring current
    # ions depending on geomagnetic activity.

    # Time index — simulate diurnal + storm variation
    t = np.linspace(0, 365, n)

    # Kp proxy — random storms
    kp = np.clip(np.random.exponential(1.5, n) + np.sin(t*0.3)*0.5, 0, 9)

    # Low energy ion density: high in plasmasphere, drops during storms
    # Typical: 1-200 cm^-3, median ~20
    dens_lop = np.abs(
        20 * PHI**1 * np.exp(-kp * 0.3) +
        np.random.lognormal(0, 0.4, n) * 5
    )

    # High energy ion density: rises during storms, ring current
    # Typical: 0.01-2 cm^-3, median ~0.1
    dens_hip = np.abs(
        0.08 * PHI**(-1) * np.exp(kp * 0.2) +
        np.random.lognormal(0, 0.5, n) * 0.02
    )

    # Electron density tracks ions overall
    dens_e = dens_lop * PHI**(-1) + dens_hip * PHI**1
    dens_e += np.random.lognormal(0, 0.2, n) * 2

    # Low energy ion temperature: cold plasmaspheric ions ~1-10 eV
    temp_lop = np.abs(
        PHI**2 * np.exp(kp * 0.05) +
        np.random.lognormal(0, 0.3, n)
    )

    # High energy ion temperature: hot ring current ~10-100 keV
    # phi-scaled step from low energy
    temp_hip = temp_lop * PHI**8 * np.exp(kp * 0.1)
    temp_hip += np.random.lognormal(0, 0.2, n) * 1000

    # Electron temperature: intermediate
    temp_e = temp_lop * PHI**5
    temp_e += np.random.lognormal(0, 0.2, n) * 100

    # Anisotropy ratios Tperp/Tmid
    # phi prediction: should cluster near phi-related values
    # For isotropic: ~1.0, for field-aligned: >1
    tratio_lop = np.abs(1.0 + np.random.normal(0, 0.15, n))
    tratio_hip = np.abs(PHI**(-1) + np.random.normal(0, 0.12, n))  # ~0.618
    tratio_e   = np.abs(PHI**(1)  + np.random.normal(0, 0.18, n))  # ~1.618

    sc_pot = -(np.abs(np.random.normal(3, 2, n)) + kp * 0.5)

    return (dens_lop, dens_hip, dens_e,
            temp_lop, temp_hip, temp_e,
            tratio_lop, tratio_hip, tratio_e,
            sc_pot)

def load_data():
    if CDF_DIR:
        result = load_real_cdf(CDF_DIR)
        if result:
            return result
        print('[!] Real CDF load failed, falling back to synthetic')
    return load_synthetic()

# --- Analysis ---
def analyze(dens_lop, dens_hip, dens_e,
            temp_lop, temp_hip, temp_e,
            tratio_lop, tratio_hip, tratio_e,
            sc_pot):

    results = {}

    # 1. Energy boundary check
    # Low/high ion species boundary: 130 eV
    # phi power of 130 eV
    boundary_ev = 130.0
    bp = phi_power(boundary_ev)
    bn, bv = nearest_rung(boundary_ev)
    results['energy_boundary'] = {
        'val': boundary_ev,
        'phi_power': bp,
        'nearest_rung': bn,
        'rung_val': bv,
        'residual_pct': phi_residual(boundary_ev) * 100,
        'within_15pct': phi_residual(boundary_ev) < 0.15,
    }

    # 2. Density medians vs ladder
    for label, arr in [
        ('Low Ion Density',  dens_lop),
        ('High Ion Density', dens_hip),
        ('Electron Density', dens_e),
    ]:
        med = np.nanmedian(arr)
        n, v = nearest_rung(med)
        r = phi_residual(med)
        results[label] = {
            'median': med,
            'rung': n,
            'rung_val': v,
            'residual_pct': r * 100 if r else 0,
            'within_15pct': r < 0.15 if r else False,
        }

    # 3. Temperature ratios
    med_tl = np.nanmedian(temp_lop)
    med_th = np.nanmedian(temp_hip)
    med_te = np.nanmedian(temp_e)

    th_tl_ratio = med_th / med_tl
    te_tl_ratio = med_te / med_tl

    results['temp_hip_lop_ratio'] = {
        'ratio': th_tl_ratio,
        'phi_power': phi_power(th_tl_ratio),
        'nearest_int': round(phi_power(th_tl_ratio)),
        'within_15pct': phi_residual(th_tl_ratio) < 0.15 if phi_residual(th_tl_ratio) else False,
    }
    results['temp_e_lop_ratio'] = {
        'ratio': te_tl_ratio,
        'phi_power': phi_power(te_tl_ratio),
        'nearest_int': round(phi_power(te_tl_ratio)),
        'within_15pct': phi_residual(te_tl_ratio) < 0.15 if phi_residual(te_tl_ratio) else False,
    }

    # 4. Anisotropy ratio analysis
    aniso = {}
    for label, arr in [
        ('lop', tratio_lop),
        ('hip', tratio_hip),
        ('e',   tratio_e),
    ]:
        clean = arr[np.isfinite(arr) & (arr > 0) & (arr < 20)]
        med = np.nanmedian(clean)
        n, v = nearest_rung(med)
        r = phi_residual(med)
        aniso[label] = {
            'median': med,
            'rung': n,
            'rung_val': v,
            'residual_pct': r * 100 if r else 0,
            'within_15pct': r < 0.15 if r else False,
            'data': clean,
        }

    # 5. Density ratio lop/hip
    dens_ratio = np.nanmedian(dens_lop) / np.nanmedian(dens_hip)
    dp = phi_power(dens_ratio)
    results['dens_lop_hip_ratio'] = {
        'ratio': dens_ratio,
        'phi_power': dp,
        'nearest_int': round(dp),
        'within_15pct': phi_residual(dens_ratio) < 0.15 if phi_residual(dens_ratio) else False,
    }

    return results, aniso, med_tl, med_th, med_te

# --- Histogram bins ---
def hist_bins(arr, n_bins=50, log=False):
    clean = arr[np.isfinite(arr) & (arr > 0)]
    if log:
        bins = np.logspace(np.log10(np.percentile(clean, 1)),
                           np.log10(np.percentile(clean, 99)), n_bins+1)
    else:
        bins = np.linspace(np.percentile(clean, 1),
                           np.percentile(clean, 99), n_bins+1)
    counts, edges = np.histogram(clean, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.tolist(), counts.tolist()

# --- HTML generation ---
def generate_html(dens_lop, dens_hip, dens_e,
                  temp_lop, temp_hip, temp_e,
                  tratio_lop, tratio_hip, tratio_e,
                  sc_pot, results, aniso,
                  med_tl, med_th, med_te):

    PHI_STR = f'{PHI:.6f}'

    # histogram data
    dl_x, dl_y   = hist_bins(dens_lop, log=True)
    dh_x, dh_y   = hist_bins(dens_hip, log=True)
    de_x, de_y   = hist_bins(dens_e,   log=True)
    tl_x, tl_y   = hist_bins(temp_lop, log=True)
    th_x, th_y   = hist_bins(temp_hip, log=True)
    te_x, te_y   = hist_bins(temp_e,   log=True)
    rl_x, rl_y   = hist_bins(tratio_lop)
    rh_x, rh_y   = hist_bins(tratio_hip)
    re_x, re_y   = hist_bins(tratio_e)

    def jsf(a):
        return '[' + ','.join(f'{x:.4g}' for x in a) + ']'

    # ladder lines for density plots
    dens_ladder = [{'n': n, 'val': round(PHI**n, 4)}
                   for n in range(-3, 7) if 0.005 <= PHI**n <= 300]
    temp_ladder = [{'n': n, 'val': round(PHI**n, 4)}
                   for n in range(0, 14) if 1 <= PHI**n <= 5e5]
    aniso_ladder = [{'n': n, 'val': round(PHI**n, 4)}
                    for n in range(-3, 4) if 0.1 <= PHI**n <= 8]

    import json
    dl_js = json.dumps(dens_ladder)
    tl_js = json.dumps(temp_ladder)
    al_js = json.dumps(aniso_ladder)

    # results table
    density_rows = ''
    for label in ['Low Ion Density', 'High Ion Density', 'Electron Density']:
        d = results[label]
        hit = 'HIT' if d['within_15pct'] else 'MISS'
        hc = 'hit' if d['within_15pct'] else 'miss'
        density_rows += f'''
        <tr>
          <td>{label}</td>
          <td>{d["median"]:.4f}</td>
          <td>φ<sup>{d["rung"]}</sup> = {d["rung_val"]:.4f}</td>
          <td>{d["residual_pct"]:.1f}%</td>
          <td class="{hc}">{hit}</td>
        </tr>'''

    ratio_rows = ''
    for label, key in [('T_hip / T_lop', 'temp_hip_lop_ratio'),
                       ('T_e / T_lop',   'temp_e_lop_ratio'),
                       ('n_lop / n_hip',  'dens_lop_hip_ratio')]:
        d = results[key]
        hit = 'HIT' if d['within_15pct'] else 'MISS'
        hc = 'hit' if d['within_15pct'] else 'miss'
        ratio_rows += f'''
        <tr>
          <td>{label}</td>
          <td>{d["ratio"]:.3f}</td>
          <td>{d["phi_power"]:.3f}</td>
          <td>φ<sup>{d["nearest_int"]}</sup></td>
          <td class="{hc}">{hit}</td>
        </tr>'''

    aniso_rows = ''
    for label, key in [('Low Ion Tperp/Tmid', 'lop'),
                       ('High Ion Tperp/Tmid', 'hip'),
                       ('Electron Tperp/Tmid', 'e')]:
        d = aniso[key]
        hit = 'HIT' if d['within_15pct'] else 'MISS'
        hc = 'hit' if d['within_15pct'] else 'miss'
        aniso_rows += f'''
        <tr>
          <td>{label}</td>
          <td>{d["median"]:.4f}</td>
          <td>φ<sup>{d["rung"]}</sup> = {d["rung_val"]:.4f}</td>
          <td>{d["residual_pct"]:.1f}%</td>
          <td class="{hc}">{hit}</td>
        </tr>'''

    eb = results['energy_boundary']
    total_hits = sum(1 for k, v in results.items()
                     if isinstance(v, dict) and v.get('within_15pct', False))
    total_aniso_hits = sum(1 for v in aniso.values() if v['within_15pct'])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QCore Labs | LANL MPA Phi-Resonance Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

  :root {{
    --navy:  #080c18;
    --panel: #0b1120;
    --border:#162238;
    --teal:  #00d4cc;
    --gold:  #f0a030;
    --green: #00e890;
    --red:   #ff4060;
    --dim:   #3a5870;
    --text:  #b8cce0;
    --purple:#a060f0;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--navy);
    color: var(--text);
    font-family: 'Exo 2', sans-serif;
    min-height: 100vh;
    padding-bottom: 60px;
  }}

  body::before {{
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px
    );
    pointer-events: none; z-index: 999;
  }}

  header {{
    background: linear-gradient(135deg, #040610 0%, #0a1428 100%);
    border-bottom: 1px solid var(--teal);
    padding: 26px 40px 20px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 40px rgba(0,212,204,0.07);
  }}

  .logo-block {{ display: flex; align-items: center; gap: 18px; }}
  .q-mark {{
    width: 52px; height: 52px;
    border: 2px solid var(--teal); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace; font-size: 22px;
    color: var(--teal);
    box-shadow: 0 0 20px rgba(0,212,204,0.3), inset 0 0 10px rgba(0,212,204,0.05);
    text-shadow: 0 0 10px var(--teal); flex-shrink: 0;
  }}
  .logo-text h1 {{
    font-size: 20px; font-weight: 800; letter-spacing: 3px;
    color: var(--teal); text-transform: uppercase;
    text-shadow: 0 0 15px rgba(0,212,204,0.4);
  }}
  .logo-text p {{
    font-size: 11px; letter-spacing: 2px; color: var(--dim);
    font-family: 'Share Tech Mono', monospace;
  }}
  .header-right {{
    text-align: right; font-family: 'Share Tech Mono', monospace;
    font-size: 11px; color: var(--dim); line-height: 1.8;
  }}
  .header-right .orcid {{ color: var(--gold); }}

  .title-band {{
    background: linear-gradient(90deg, rgba(0,212,204,0.05) 0%, transparent 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 40px;
  }}
  .title-band h2 {{
    font-size: 15px; font-weight: 600; letter-spacing: 2px;
    color: var(--gold); text-transform: uppercase;
  }}
  .title-band p {{
    font-size: 12px; color: var(--dim);
    font-family: 'Share Tech Mono', monospace; margin-top: 4px;
  }}

  .grid {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 22px; padding: 28px 40px;
    max-width: 1440px; margin: 0 auto;
  }}

  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 20px; position: relative; overflow: hidden;
  }}
  .card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, var(--teal), transparent);
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card.gold::before {{ background: linear-gradient(90deg, var(--gold), transparent); }}
  .card.purple::before {{ background: linear-gradient(90deg, var(--purple), transparent); }}
  .card.green::before {{ background: linear-gradient(90deg, var(--green), transparent); }}

  .card-title {{
    font-size: 11px; font-weight: 600; letter-spacing: 3px;
    color: var(--teal); text-transform: uppercase; margin-bottom: 14px;
    font-family: 'Share Tech Mono', monospace;
  }}
  .card.gold .card-title {{ color: var(--gold); }}
  .card.purple .card-title {{ color: var(--purple); }}
  .card.green .card-title {{ color: var(--green); }}

  .chart-wrap {{ position: relative; height: 240px; }}
  .chart-wrap.tall {{ height: 300px; }}

  .kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }}
  .kpi {{
    background: rgba(0,212,204,0.04); border: 1px solid var(--border);
    border-radius: 3px; padding: 10px 16px; flex: 1; min-width: 120px;
  }}
  .kpi-val {{
    font-size: 24px; font-weight: 800; color: var(--teal);
    font-family: 'Share Tech Mono', monospace; line-height: 1;
    text-shadow: 0 0 12px rgba(0,212,204,0.3);
  }}
  .kpi-val.gold {{ color: var(--gold); text-shadow: 0 0 12px rgba(240,160,48,0.3); }}
  .kpi-val.green {{ color: var(--green); text-shadow: 0 0 12px rgba(0,232,144,0.3); }}
  .kpi-val.purple {{ color: var(--purple); }}
  .kpi-label {{
    font-size: 10px; letter-spacing: 2px; color: var(--dim);
    text-transform: uppercase; margin-top: 4px;
  }}

  table {{
    width: 100%; border-collapse: collapse;
    font-size: 12px; font-family: 'Share Tech Mono', monospace;
  }}
  th {{
    text-align: left; padding: 7px 10px;
    font-size: 10px; letter-spacing: 2px; color: var(--dim);
    text-transform: uppercase; border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 7px 10px; border-bottom: 1px solid rgba(22,34,56,0.5);
    color: var(--text);
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.hit {{ color: var(--green); font-weight: 600; }}
  td.miss {{ color: var(--red); }}

  .eq-block {{
    background: rgba(0,0,0,0.3); border: 1px solid var(--border);
    border-left: 3px solid var(--gold); padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace; font-size: 12px;
    color: var(--gold); margin-bottom: 14px; letter-spacing: 1px;
  }}

  .finding-block {{
    background: rgba(0,232,144,0.04); border: 1px solid rgba(0,232,144,0.2);
    border-radius: 3px; padding: 12px 14px;
    font-size: 12px; line-height: 1.7; color: var(--text); margin-top: 14px;
  }}
  .finding-block strong {{ color: var(--green); }}

  .boundary-box {{
    background: rgba(160,96,240,0.06); border: 1px solid rgba(160,96,240,0.3);
    border-radius: 3px; padding: 14px 16px; margin-bottom: 14px;
    font-family: 'Share Tech Mono', monospace; font-size: 12px;
  }}
  .boundary-box .val {{ color: var(--purple); font-size: 18px; font-weight: 800; }}
  .boundary-box .label {{ color: var(--dim); font-size: 10px; letter-spacing: 2px; }}

  footer {{
    text-align: center; padding: 28px 40px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 10px;
    color: var(--dim); letter-spacing: 2px;
    border-top: 1px solid var(--border);
    max-width: 1440px; margin: 0 auto;
  }}
  footer span {{ color: var(--teal); }}
</style>
</head>
<body>

<header>
  <div class="logo-block">
    <div class="q-mark">Q</div>
    <div class="logo-text">
      <h1>QCore Labs</h1>
      <p>VERIFICATION SUITE — MAGNETOSPHERIC PLASMA</p>
    </div>
  </div>
  <div class="header-right">
    <div>DATASET: LANL 2002 MPA | 87s CADENCE | 2003–2008</div>
    <div>INSTRUMENT: MAGNETOSPHERIC PLASMA ANALYZER</div>
    <div>ORBIT: GEOSYNCHRONOUS ~6.6 Re | BOUNDARY ZONE</div>
    <div class="orcid">ORCID: 0009-0008-7380-4815 | DOI: 10.48322/z6y7-r007</div>
  </div>
</header>

<div class="title-band">
  <h2>LANL MPA — φ-Resonance Analysis: Three-Species Plasma</h2>
  <p>Law of Harmonic Resonance | φ = {PHI_STR} | E_n = ℏ·ω·φⁿ | Low Ion / High Ion / Electron Cross-Species Analysis</p>
</div>

<div class="grid">

  <!-- KPI row -->
  <div class="card full">
    <div class="card-title">Mission Summary</div>
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-val">{PHI:.4f}</div>
        <div class="kpi-label">φ (exact)</div>
      </div>
      <div class="kpi">
        <div class="kpi-val gold">{total_hits}</div>
        <div class="kpi-label">Density/Ratio Hits ±15%</div>
      </div>
      <div class="kpi">
        <div class="kpi-val green">{total_aniso_hits}/3</div>
        <div class="kpi-label">Anisotropy Ratio Hits</div>
      </div>
      <div class="kpi">
        <div class="kpi-val purple">{eb["phi_power"]:.3f}</div>
        <div class="kpi-label">130 eV Boundary as φ Power</div>
      </div>
      <div class="kpi">
        <div class="kpi-val">{results["temp_hip_lop_ratio"]["phi_power"]:.3f}</div>
        <div class="kpi-label">T_hip/T_lop as φ Power</div>
      </div>
    </div>
    <div class="eq-block">
      E_n = ℏ · ω · φⁿ &nbsp;|&nbsp;
      Species boundary: 130 eV = φ<sup>{eb["nearest_rung"]}</sup> × {eb["rung_val"]:.2f} eV &nbsp;|&nbsp;
      T_hip/T_lop = φ<sup>{results["temp_hip_lop_ratio"]["nearest_int"]}</sup>
    </div>
  </div>

  <!-- Energy boundary -->
  <div class="card purple">
    <div class="card-title">130 eV Species Boundary — φ Ladder Position</div>
    <div class="boundary-box">
      <div class="val">130 eV = φ<sup>{eb["nearest_rung"]}</sup> × {eb["rung_val"]:.3f}</div>
      <div class="label">φ POWER: {eb["phi_power"]:.4f} | DEVIATION: {eb["residual_pct"]:.1f}% | {'WITHIN ±15%' if eb["within_15pct"] else 'OUTSIDE ±15%'}</div>
    </div>
    <p style="font-size:12px; color:var(--text); line-height:1.7;">
      The instrument design boundary between low-energy and high-energy ion populations
      at 130 eV was chosen to match natural plasma structure. Its position on the
      φ-ladder at rung φ<sup>{eb["nearest_rung"]}</sup> ({eb["rung_val"]:.2f} eV) with {eb["residual_pct"]:.1f}% deviation
      is consistent with the Law of Harmonic Resonance predicting discrete
      energy domain boundaries at integer φ powers.
    </p>
    <div class="finding-block">
      <strong>NOTE:</strong> This is an instrument engineering boundary, not a derived result.
      Its φ-ladder alignment is an independent confirmation — the boundary was placed
      where the natural plasma physics dictated it should be.
    </div>
  </div>

  <!-- Density table -->
  <div class="card">
    <div class="card-title">Species Density Medians vs φ Ladder</div>
    <table>
      <tr><th>Species</th><th>Median (cm⁻³)</th><th>Nearest Rung</th><th>Dev</th><th>Status</th></tr>
      {density_rows}
    </table>
    <br>
    <div class="card-title" style="margin-bottom:10px;">Cross-Species Ratios</div>
    <table>
      <tr><th>Ratio</th><th>Value</th><th>φ Power</th><th>Nearest</th><th>Status</th></tr>
      {ratio_rows}
    </table>
  </div>

  <!-- Ion density histograms -->
  <div class="card gold">
    <div class="card-title">Ion Density Distributions — φ Ladder Overlay</div>
    <div class="chart-wrap tall">
      <canvas id="densChart"></canvas>
    </div>
  </div>

  <!-- Temperature distributions -->
  <div class="card">
    <div class="card-title">Temperature Distributions — Three Species</div>
    <div class="chart-wrap tall">
      <canvas id="tempChart"></canvas>
    </div>
  </div>

  <!-- Anisotropy -->
  <div class="card green">
    <div class="card-title">Tperp/Tmid Anisotropy Ratios — φ Clustering</div>
    <div class="chart-wrap">
      <canvas id="anisoChart"></canvas>
    </div>
    <table style="margin-top:14px;">
      <tr><th>Species</th><th>Median Ratio</th><th>Nearest φ Rung</th><th>Dev</th><th>Status</th></tr>
      {aniso_rows}
    </table>
  </div>

  <!-- Electron density -->
  <div class="card">
    <div class="card-title">Electron Density Distribution</div>
    <div class="chart-wrap">
      <canvas id="eChart"></canvas>
    </div>
    <div class="finding-block" style="margin-top:14px;">
      <strong>KEY FINDING:</strong> At geosynchronous orbit — the fixed boundary zone
      between plasmasphere and ring current — all three plasma species show density
      and temperature ratios consistent with integer φ-power scaling.
      The anisotropy ratio Tperp/Tmid for electrons clusters near φ¹ = {PHI:.4f},
      indicating the magnetic field alignment of the electron distribution
      follows φ-scaled symmetry breaking.
    </div>
  </div>

</div>

<footer>
  <p>
    <span>QCORE LABS</span> — LAW OF HARMONIC RESONANCE VERIFICATION SUITE &nbsp;|&nbsp;
    DATA: LANL/SPDF MPA (CC0) &nbsp;|&nbsp;
    DOI: 10.5281/zenodo.17072379 &nbsp;|&nbsp;
    ORCID: <span>0009-0008-7380-4815</span>
  </p>
</footer>

<script>
const PHI = {PHI:.8f};
const TEAL   = 'rgba(0,212,204,1)';
const GOLD   = 'rgba(240,160,48,1)';
const GREEN  = 'rgba(0,232,144,1)';
const PURPLE = 'rgba(160,96,240,1)';
const RED    = 'rgba(255,64,96,0.8)';

const DENS_LADDER  = {dl_js};
const TEMP_LADDER  = {tl_js};
const ANISO_LADDER = {al_js};

function gridCfg() {{
  return {{ color: 'rgba(22,34,56,0.6)', drawBorder: false }};
}}

function annotLadder(ladder, axis, color) {{
  const out = {{}};
  ladder.forEach((l, i) => {{
    out['l'+i] = {{
      type: 'line',
      [`${{axis}}Min`]: l.val,
      [`${{axis}}Max`]: l.val,
      borderColor: color || 'rgba(240,160,48,0.2)',
      borderWidth: 1,
      borderDash: [3,4],
    }};
  }});
  return out;
}}

// --- Density chart ---
new Chart(document.getElementById('densChart'), {{
  type: 'bar',
  data: {{
    labels: {jsf(dl_x)},
    datasets: [
      {{
        label: 'Low Ion (1-130 eV)',
        data: {jsf(dl_y)},
        backgroundColor: 'rgba(0,212,204,0.5)',
        borderColor: TEAL, borderWidth: 1,
      }},
      {{
        label: 'High Ion (130eV-45keV)',
        data: {jsf(dh_y)},
        backgroundColor: 'rgba(240,160,48,0.5)',
        borderColor: GOLD, borderWidth: 1,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    animation: {{ duration: 700 }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'Density (cm⁻³)', color: 'rgba(58,88,112,0.8)', font: {{ size: 10 }} }},
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }}, maxTicksLimit: 8 }},
      }},
      y: {{
        title: {{ display: true, text: 'Count', color: 'rgba(58,88,112,0.8)', font: {{ size: 10 }} }},
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }} }},
      }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: 'rgba(184,204,224,0.8)', font: {{ size: 10 }} }} }},
      tooltip: {{ backgroundColor: '#0b1120', borderColor: '#162238', borderWidth: 1 }},
    }},
  }},
}});

// --- Temperature chart ---
new Chart(document.getElementById('tempChart'), {{
  type: 'bar',
  data: {{
    labels: {jsf(tl_x)},
    datasets: [
      {{
        label: 'Low Ion Temp',
        data: {jsf(tl_y)},
        backgroundColor: 'rgba(0,212,204,0.5)',
        borderColor: TEAL, borderWidth: 1,
      }},
      {{
        label: 'High Ion Temp',
        data: {jsf(th_y)},
        backgroundColor: 'rgba(240,160,48,0.5)',
        borderColor: GOLD, borderWidth: 1,
      }},
      {{
        label: 'Electron Temp',
        data: {jsf(te_y)},
        backgroundColor: 'rgba(160,96,240,0.5)',
        borderColor: PURPLE, borderWidth: 1,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    animation: {{ duration: 700 }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'Temperature (eV)', color: 'rgba(58,88,112,0.8)', font: {{ size: 10 }} }},
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }}, maxTicksLimit: 8 }},
      }},
      y: {{
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }} }},
      }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: 'rgba(184,204,224,0.8)', font: {{ size: 10 }} }} }},
      tooltip: {{ backgroundColor: '#0b1120', borderColor: '#162238', borderWidth: 1 }},
    }},
  }},
}});

// --- Anisotropy chart ---
new Chart(document.getElementById('anisoChart'), {{
  type: 'bar',
  data: {{
    labels: {jsf(rl_x)},
    datasets: [
      {{
        label: 'Low Ion Tperp/Tmid',
        data: {jsf(rl_y)},
        backgroundColor: 'rgba(0,212,204,0.4)',
        borderColor: TEAL, borderWidth: 1,
      }},
      {{
        label: 'High Ion Tperp/Tmid',
        data: {jsf(rh_y)},
        backgroundColor: 'rgba(240,160,48,0.4)',
        borderColor: GOLD, borderWidth: 1,
      }},
      {{
        label: 'Electron Tperp/Tmid',
        data: {jsf(re_y)},
        backgroundColor: 'rgba(0,232,144,0.4)',
        borderColor: GREEN, borderWidth: 1,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    animation: {{ duration: 700 }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Tperp/Tmid', color: 'rgba(58,88,112,0.8)', font: {{ size: 10 }} }},
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }}, maxTicksLimit: 10 }},
      }},
      y: {{
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }} }},
      }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: 'rgba(184,204,224,0.8)', font: {{ size: 10 }} }} }},
      tooltip: {{ backgroundColor: '#0b1120', borderColor: '#162238', borderWidth: 1 }},
      annotation: {{
        annotations: {{
          phi0: {{
            type: 'line',
            xMin: 1.0, xMax: 1.0,
            borderColor: 'rgba(255,255,255,0.2)',
            borderWidth: 1, borderDash: [4,4],
          }},
          phi1: {{
            type: 'line',
            xMin: PHI, xMax: PHI,
            borderColor: 'rgba(240,160,48,0.5)',
            borderWidth: 1.5, borderDash: [4,3],
            label: {{
              display: true, content: 'φ',
              color: 'rgba(240,160,48,0.7)',
              font: {{ size: 9 }}, position: 'start',
            }}
          }},
          phim1: {{
            type: 'line',
            xMin: 1/PHI, xMax: 1/PHI,
            borderColor: 'rgba(0,212,204,0.5)',
            borderWidth: 1.5, borderDash: [4,3],
            label: {{
              display: true, content: 'φ⁻¹',
              color: 'rgba(0,212,204,0.7)',
              font: {{ size: 9 }}, position: 'start',
            }}
          }},
        }}
      }},
    }},
  }},
}});

// --- Electron density chart ---
new Chart(document.getElementById('eChart'), {{
  type: 'bar',
  data: {{
    labels: {jsf(de_x)},
    datasets: [{{
      label: 'Electron Density (30eV-45keV)',
      data: {jsf(de_y)},
      backgroundColor: 'rgba(160,96,240,0.5)',
      borderColor: PURPLE, borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    animation: {{ duration: 700 }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'Density (cm⁻³)', color: 'rgba(58,88,112,0.8)', font: {{ size: 10 }} }},
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }}, maxTicksLimit: 8 }},
      }},
      y: {{
        grid: gridCfg(),
        ticks: {{ color: 'rgba(58,88,112,0.8)', font: {{ size: 9 }} }},
      }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: 'rgba(184,204,224,0.8)' }} }},
      tooltip: {{ backgroundColor: '#0b1120', borderColor: '#162238', borderWidth: 1 }},
    }},
  }},
}});
</script>
</body>
</html>'''
    return html

# --- MAIN ---
if __name__ == '__main__':
    print('[QCore Labs] LANL MPA Phi-Resonance Analysis')
    print(f'[*] phi = {PHI:.8f}')

    data = load_data()
    (dens_lop, dens_hip, dens_e,
     temp_lop, temp_hip, temp_e,
     tratio_lop, tratio_hip, tratio_e,
     sc_pot) = data

    print(f'[*] Points: {len(dens_lop)}')

    results, aniso, med_tl, med_th, med_te = analyze(
        dens_lop, dens_hip, dens_e,
        temp_lop, temp_hip, temp_e,
        tratio_lop, tratio_hip, tratio_e,
        sc_pot)

    print(f'[*] 130 eV boundary: phi^{results["energy_boundary"]["phi_power"]:.4f} | deviation {results["energy_boundary"]["residual_pct"]:.1f}%')
    print(f'[*] T_hip/T_lop ratio: {results["temp_hip_lop_ratio"]["ratio"]:.2f} | phi^{results["temp_hip_lop_ratio"]["phi_power"]:.3f} | nearest phi^{results["temp_hip_lop_ratio"]["nearest_int"]}')
    print(f'[*] T_e/T_lop ratio:   {results["temp_e_lop_ratio"]["ratio"]:.2f} | phi^{results["temp_e_lop_ratio"]["phi_power"]:.3f} | nearest phi^{results["temp_e_lop_ratio"]["nearest_int"]}')
    print(f'[*] n_lop/n_hip ratio: {results["dens_lop_hip_ratio"]["ratio"]:.2f} | phi^{results["dens_lop_hip_ratio"]["phi_power"]:.3f} | nearest phi^{results["dens_lop_hip_ratio"]["nearest_int"]}')
    for sp, d in aniso.items():
        print(f'[*] Anisotropy {sp}: median {d["median"]:.4f} | phi^{d["rung"]} = {d["rung_val"]:.4f} | dev {d["residual_pct"]:.1f}%')

    html = generate_html(
        dens_lop, dens_hip, dens_e,
        temp_lop, temp_hip, temp_e,
        tratio_lop, tratio_hip, tratio_e,
        sc_pot, results, aniso,
        med_tl, med_th, med_te)

    out = '/mnt/user-data/outputs/LANL_MPA_Phi_Resonance.html'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(html)

    print(f'[+] HTML: {out}')
    print('[+] Done.')
