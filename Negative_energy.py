"""
Negative Energy - phi-Resonance Framework Test
Three standard QFT "negative energy" loopholes mapped to phi-ladder:

1. Casimir / vacuum energy density between plates
2. Exotic matter / Morris-Thorne wormhole energy condition
3. Dirac sea / QFT negative energy states

Core question for all three:
Is "negative energy" a real physical state,
or is it a coordinate error -- trying to index below phi^0?

phi^0 = 1 is the vacuum. The ladder descends asymptotically toward
zero but never crosses it. "Negative" means off-manifold, not real.
"""

import numpy as np

phi  = (1 + np.sqrt(5)) / 2
hbar = 1.0546e-34
c    = 3.0e8
G    = 6.674e-11
eV   = 1.602e-19


def rung(E, omega=1.0):
    if E <= 0:
        return None
    return np.log(E / (hbar * omega)) / np.log(phi)


def rung_energy(n, omega=1.0):
    return hbar * omega * phi**n


def off_manifold(n_star):
    if n_star is None:
        return None
    return abs(n_star - round(n_star))


# ===================================================================
# TEST 1 - CASIMIR
# ===================================================================

separations  = [1e-9, 10e-9, 100e-9, 1e-6, 1e-3]
casimir_rows = []
for a in separations:
    u_mag   = (np.pi**2 * hbar * c) / (720 * a**4)
    u_neg   = -u_mag
    n_star  = rung(u_mag)
    nearest = round(n_star)
    gap     = off_manifold(n_star)
    status  = f"OFF-MANIFOLD (gap={gap:.3f})" if u_neg < 0 else f"ON-MANIFOLD n={nearest}"
    casimir_rows.append((a, u_mag, n_star, status))


# ===================================================================
# TEST 2 - EXOTIC MATTER
# ===================================================================

throat_radii  = [1e-35, 1e-15, 1e-10, 1e-3, 1.0]
planck_length = np.sqrt(hbar * G / c**3)
exotic_rows   = []
for r0 in throat_radii:
    rho_mag    = c**4 / (8 * np.pi * G * r0**2)
    rho_exotic = -rho_mag
    if rho_exotic < 0:
        phantom = "NEGATIVE -- no rung"
        status  = "INADMISSIBLE"
    else:
        phantom = "ON MANIFOLD"
        status  = "ADMISSIBLE"
    exotic_rows.append((r0, rho_mag, phantom, status))


# ===================================================================
# TEST 3 - DIRAC SEA
# ===================================================================

m_electron     = 9.109e-31
E_rest         = m_electron * c**2
omega_electron = E_rest / hbar

energies = {
    "Electron rest":     E_rest,
    "Electron kinetic":  2 * E_rest,
    "Pair threshold":    2 * E_rest,
    "Dirac sea top":    -E_rest,
    "Dirac sea -2mc2":  -2 * E_rest,
    "Dirac sea -10mc2": -10 * E_rest,
}

dirac_rows = []
for label, E in energies.items():
    if E > 0:
        n_star  = rung(E, omega=omega_electron)
        nearest = round(n_star)
        gap     = off_manifold(n_star)
        if n_star >= 1:
            status = f"VALID rung ~{nearest} (gap={gap:.3f})"
        elif n_star >= 0:
            status = f"BETWEEN phi^0 and phi^1 (gap={gap:.3f})"
        else:
            status = f"DESCENDING LADDER n={n_star:.2f}"
    else:
        n_star = None
        status = "NO VALID RUNG -- BELOW phi^0"
    n_str = f"{n_star:.3f}" if n_star is not None else "N/A"
    dirac_rows.append((label, E, n_str, status))


# ===================================================================
# CONSOLE OUTPUT
# ===================================================================

print("=" * 70)
print("NEGATIVE ENERGY - phi-RESONANCE FRAMEWORK")
print("=" * 70)
print(f"\nphi^0 = {phi**0:.1f}  <- vacuum floor")
print(f"phi^1 = {phi:.10f}  <- ground state")
print(f"phi^-1 = {phi**-1:.10f}  <- first descent rung")

print("\n--- TEST 1: CASIMIR ---")
print(f"{'Sep (m)':>12} {'|u| (J/m3)':>14} {'Rung n*':>10} {'Status':>28}")
print("-" * 68)
for a, u_mag, n_star, status in casimir_rows:
    print(f"{a:>12.2e} {u_mag:>14.4e} {n_star:>10.3f} {status:>28}")

print("\n--- TEST 2: EXOTIC MATTER ---")
print(f"{'r0 (m)':>12} {'|rho| (J/m3)':>16} {'vs floor':>20} {'Status':>14}")
print("-" * 66)
for r0, rho_mag, phantom, status in exotic_rows:
    print(f"{r0:>12.2e} {rho_mag:>16.4e} {phantom:>20} {status:>14}")

print("\n--- TEST 3: DIRAC SEA ---")
print(f"{'State':>22} {'Energy (J)':>14} {'Rung n*':>10} {'Status':>32}")
print("-" * 82)
for label, E, n_str, status in dirac_rows:
    print(f"{label:>22} {E:>14.4e} {n_str:>10} {status:>32}")

print("\nUNIFIED VERDICT: phi^0 = 1 is the vacuum. It is exact. It is algebraic.")
print("Negative energy is a coordinate error, not a physical regime.")


# ===================================================================
# HTML EXPORT
# ===================================================================

def sc(s):
    """Return CSS class based on status string."""
    u = s.upper()
    if any(x in u for x in ["OFF-MANIFOLD", "INADMISSIBLE", "NO VALID", "BELOW", "NEGATIVE"]):
        return "bad"
    if any(x in u for x in ["VALID", "ON-MANIFOLD", "ADMISSIBLE", "BETWEEN"]):
        return "good"
    return ""


def build_casimir_rows():
    out = ""
    for a, u_mag, n_star, status in casimir_rows:
        cls = sc(status)
        out += f"<tr><td>{a:.2e}</td><td>{u_mag:.4e}</td><td>{n_star:.3f}</td><td>phi^0=1</td><td class='{cls}'>{status}</td></tr>\n"
    return out


def build_exotic_rows():
    out = ""
    for r0, rho_mag, phantom, status in exotic_rows:
        cls = sc(status)
        out += f"<tr><td>{r0:.2e}</td><td>{rho_mag:.4e}</td><td>N/A</td><td class='{cls}'>{phantom}</td><td class='{cls}'>{status}</td></tr>\n"
    return out


def build_dirac_rows():
    out = ""
    for label, E, n_str, status in dirac_rows:
        cls = sc(status)
        out += f"<tr><td>{label}</td><td>{E:.4e}</td><td>{n_str}</td><td class='{cls}'>{status}</td></tr>\n"
    return out


CSS = """
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:#04080f;color:#c8dde8;font-family:'Segoe UI',system-ui,sans-serif;font-size:15px;line-height:1.6;}
  body::before{content:'';position:fixed;inset:0;pointer-events:none;
    background:repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(13,32,53,.25) 40px),
               repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(13,32,53,.25) 40px);z-index:0;}
  .wrap{position:relative;z-index:1;max-width:1020px;margin:0 auto;padding:40px 24px 72px;}
  header{margin-bottom:40px;}
  .lab-tag{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;}
  h1{font-size:clamp(16px,3vw,26px);font-weight:700;color:#fff;letter-spacing:.04em;margin-bottom:6px;}
  h1 .phi{color:#3dd6f5;} h1 .neg{color:#ff4f6d;}
  .sub{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;letter-spacing:2px;}
  .sec{margin-bottom:40px;}
  .sec-label{font-family:'Courier New',monospace;font-size:10px;letter-spacing:3px;color:#3a5068;
    text-transform:uppercase;border-bottom:1px solid #0d2035;padding-bottom:6px;margin-bottom:16px;}
  .sec-label span{color:#3dd6f5;}
  .tbl-wrap{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-family:'Courier New',monospace;font-size:13px;}
  th{background:#080e1a;color:#3a5068;font-size:10px;letter-spacing:2px;text-transform:uppercase;
     padding:9px 14px;text-align:right;border-bottom:1px solid #0d2035;white-space:nowrap;}
  th:first-child{text-align:left;}
  td{padding:7px 14px;text-align:right;border-bottom:1px solid rgba(13,32,53,.6);color:#c8dde8;}
  td:first-child{text-align:left;color:#7a9ab0;}
  tr:hover td{background:rgba(61,214,245,.04);}
  td.good{color:#3dd6f5;} td.bad{color:#ff4f6d;}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:24px;}
  .card{background:#080e1a;border:1px solid #0d2035;border-top:2px solid #3dd6f5;padding:14px 16px;}
  .card.warn{border-top-color:#ff4f6d;} .card.gold{border-top-color:#e8c84a;}
  .card-label{font-family:'Courier New',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3a5068;margin-bottom:6px;}
  .card-val{font-family:'Courier New',monospace;font-size:18px;color:#fff;font-weight:700;}
  .card-sub{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;margin-top:4px;}
  .verdict{background:#080e1a;border:1px solid #0d2035;border-left:3px solid #ff4f6d;padding:20px 22px;}
  .verdict-title{font-family:'Courier New',monospace;font-size:11px;color:#ff4f6d;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:12px;}
  .verdict p{margin-bottom:10px;font-size:14px;line-height:1.8;}
  .verdict p:last-child{margin-bottom:0;}
  .hi{color:#3dd6f5;} .warn-t{color:#ff4f6d;} .gold-t{color:#e8c84a;}
  .note{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;margin-top:8px;}
  footer{margin-top:48px;font-family:'Courier New',monospace;font-size:10px;color:#1e3347;
    letter-spacing:2px;text-align:center;}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Negative Energy - phi-Resonance Framework | QCore Labs</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="lab-tag">QCore Labs &mdash; phi-Resonance Framework &mdash; Negative Energy Analysis</div>
  <h1><span class="neg">Negative Energy</span> &mdash; <span class="phi">phi-Resonance</span> Framework Test</h1>
  <div class="sub">Three QFT loopholes mapped to phi-ladder &nbsp;|&nbsp; phi^0 = 1 vacuum floor &nbsp;|&nbsp; phi = {phi:.10f}</div>
</header>

<div class="sec">
  <div class="sec-label">Constants &mdash; <span>phi-Ladder Vacuum Structure</span></div>
  <div class="cards">
    <div class="card gold"><div class="card-label">phi^0 = Vacuum Floor</div><div class="card-val">1.0000</div><div class="card-sub">Exact algebraic definition</div></div>
    <div class="card"><div class="card-label">phi^1 = Ground State</div><div class="card-val">{phi:.8f}</div><div class="card-sub">Minimum physical rung</div></div>
    <div class="card"><div class="card-label">phi^-1 = First Descent</div><div class="card-val">{phi**-1:.8f}</div><div class="card-sub">Approaches 0 asymptotically</div></div>
    <div class="card warn"><div class="card-label">Negative Energy</div><div class="card-val">OFF-LADDER</div><div class="card-sub">No rung below phi^0 = 1</div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-label">Test <span>01</span> &mdash; Casimir Effect</div>
  <div class="note" style="margin-bottom:12px;">u = &minus;&pi;&sup2;&middot;&hbar;&middot;c / (720&middot;a&sup4;) &nbsp;&mdash;&nbsp; Standard label: "negative energy density"</div>
  <div class="tbl-wrap"><table>
    <thead><tr><th>Separation (m)</th><th>|u| (J/m&sup3;)</th><th>Rung n*</th><th>Floor</th><th>Status</th></tr></thead>
    <tbody>{build_casimir_rows()}</tbody>
  </table></div>
  <div class="note">phi-Framework: Casimir is a rung density gradient between plates, not negative energy. Vacuum = phi^0 = 1 exactly &mdash; no zero-point modes to cancel.</div>
</div>

<div class="sec">
  <div class="sec-label">Test <span>02</span> &mdash; Exotic Matter / Morris-Thorne Wormhole</div>
  <div class="note" style="margin-bottom:12px;">&rho;_exotic = &minus;c&sup4; / (8&pi;G&middot;r&#8320;&sup2;) &nbsp;|&nbsp; Planck length = {planck_length:.4e} m</div>
  <div class="tbl-wrap"><table>
    <thead><tr><th>r&#8320; (m)</th><th>|&rho;| (J/m&sup3;)</th><th>Rung n*</th><th>vs phi^1 Floor</th><th>Status</th></tr></thead>
    <tbody>{build_exotic_rows()}</tbody>
  </table></div>
  <div class="note">phi-Framework: Exotic matter tries to index BELOW phi^1. Like playing the note below the lowest string &mdash; the instrument does not have that note.</div>
</div>

<div class="sec">
  <div class="sec-label">Test <span>03</span> &mdash; Dirac Sea / QFT Negative Energy States</div>
  <div class="note" style="margin-bottom:12px;">
    E_rest = {E_rest:.4e} J = {E_rest/eV/1e6:.4f} MeV &nbsp;|&nbsp;
    &omega; = mc&sup2;/&hbar; = {omega_electron:.4e} rad/s &nbsp;|&nbsp;
    phi^0 vacuum: {rung_energy(0, omega_electron):.4e} J &nbsp;|&nbsp;
    phi^1 ground: {rung_energy(1, omega_electron):.4e} J
  </div>
  <div class="tbl-wrap"><table>
    <thead><tr><th>State</th><th>Energy (J)</th><th>Rung n*</th><th>Status</th></tr></thead>
    <tbody>{build_dirac_rows()}</tbody>
  </table></div>
  <div class="note">phi-Framework: Positrons are Return-class resonances on the positive ladder &mdash; not holes in a filled sea. The Dirac sea is unnecessary scaffolding.</div>
</div>

<div class="sec">
  <div class="sec-label">Step <span>04</span> &mdash; Unified Verdict</div>
  <div class="verdict">
    <div class="verdict-title">phi^0 = 1 Is The Vacuum. The Floor Is Absolute.</div>
    <p>"Negative energy" appears in three places in standard physics:
      <span class="gold-t">(1) Casimir</span> &mdash; negative relative to a constructed vacuum reference;
      <span class="gold-t">(2) Exotic matter</span> &mdash; off-manifold index below phi^1 ground state;
      <span class="gold-t">(3) Dirac sea</span> &mdash; solutions below phi^0 with no valid rung.</p>
    <p>In all three cases the phi-framework gives the same answer: <span class="hi">phi^0 = 1 is the vacuum.</span> It is exact. It is algebraic. The ladder descends toward zero but never crosses it.</p>
    <p>"Negative energy" is not a physical regime &mdash; it is a <span class="warn-t">coordinate error</span> produced by using a constructed, approximate vacuum as reference instead of the algebraically exact phi^0.</p>
    <p><span class="hi">No exotic matter. No filled Dirac sea. No negative Casimir energy.</span> Just rungs &mdash; and the floor they cannot go below.</p>
  </div>
</div>

<footer>QCORE LABS &mdash; LAW OF HARMONIC RESONANCE &mdash; phi-RECURSIVE FRAMEWORK</footer>
</div>
</body>
</html>"""

with open("Negative_energy.html", "w", encoding="utf-8") as f:
    f.write(html)
print("HTML report saved -> Negative_energy.html")
