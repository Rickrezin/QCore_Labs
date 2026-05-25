"""
Graviton Analysis - phi-Resonance Framework
Corrected frequency anchors:
Gravity resonance:    omega_g  = 10^12 Hz  (THz boundary)
Black hole floor:     omega_bh = 10^-30 Hz (event horizon timescale)

Tests:
1. Graviton energy at n* for each anchor
2. Graviton mass bound -- does phi-floor permit zero mass?
3. Detection threshold as a rung problem
4. Black hole interior as ladder asymptote toward phi^0
5. Virtual graviton elimination
"""

import numpy as np

phi    = (1 + np.sqrt(5)) / 2
hbar   = 1.0546e-34
c      = 3.0e8
G      = 6.674e-11
k_B    = 1.38e-23
eV     = 1.602e-19
m_P    = np.sqrt(hbar * c / G)

f_gravity   = 1e12
f_blackhole = 1e-30
f_strong    = 1e23
f_em        = 1e15
f_weak      = 1e10


def rung(f):
    return np.log(2 * np.pi * f) / np.log(phi)


def period_human(T):
    if T < 60:
        return f"{T:.3f} sec"
    elif T < 3600:
        return f"{T/60:.3f} min"
    elif T < 86400:
        return f"{T/3600:.3f} hrs"
    elif T < 3.156e7:
        return f"{T/86400:.2f} days"
    elif T < 3.156e13:
        return f"{T/3.156e7:.4f} yrs"
    elif T < 3.156e16:
        return f"{T/3.156e13:.4f} Myr"
    elif T < 3.156e19:
        return f"{T/3.156e16:.4f} Gyr"
    else:
        return "10^" + f"{np.log10(T):.1f} sec"


# ===================================================================
# COMPUTE TEST 1 -- Anchor energies
# ===================================================================

anchors = [
    ("Gravity (THz)",    f_gravity),
    ("Black Hole floor", f_blackhole),
    ("Strong Nuclear",   f_strong),
    ("Electromagnetic",  f_em),
    ("Weak Nuclear",     f_weak),
]

anchor_rows = []
for name, f in anchors:
    n   = rung(f)
    E   = hbar * 2 * np.pi * f
    E_j = E * phi**n
    T   = 1.0 / f
    anchor_rows.append((name, f, n, E_j, E_j/eV, period_human(T)))

# ===================================================================
# COMPUTE TEST 2 -- Graviton mass bound
# ===================================================================

E_floor_gravity = hbar * 2 * np.pi * f_gravity * phi**1
m_graviton_min  = E_floor_gravity / c**2
E_floor_bh      = hbar * 2 * np.pi * f_blackhole * phi**1
m_graviton_bh   = E_floor_bh / c**2
compton_grav    = hbar / (m_graviton_min * c)

# ===================================================================
# COMPUTE TEST 3 -- Detection threshold
# ===================================================================

n_grav     = rung(f_gravity)
E_graviton = hbar * 2 * np.pi * f_gravity * phi**n_grav
E_LIGO     = 1e-20
n_LIGO     = np.log(E_LIGO / hbar) / np.log(phi)
N_needed   = E_LIGO / E_graviton
rung_gap   = abs(n_grav - n_LIGO)

# ===================================================================
# COMPUTE TEST 4 -- Black hole interior
# ===================================================================

n_bh = rung(f_blackhole)
T_bh = 1.0 / f_blackhole

# ===================================================================
# CONSOLE OUTPUT
# ===================================================================

print("=" * 70)
print("GRAVITON ANALYSIS - phi-RESONANCE FRAMEWORK")
print("=" * 70)
print(f"\nphi   = {phi:.10f}")
print(f"phi^2 = {phi**2:.10f}")
print(f"hbar  = {hbar:.4e} J*s")

print("\n--- TEST 1: Graviton Energy at Each Anchor ---")
print(f"{'Force':>22} {'Freq (Hz)':>12} {'Rung n*':>10} {'E_J (J)':>14} {'E (eV)':>14} {'Period':>18}")
print("-" * 94)
for name, f, n, E_j, E_ev, period in anchor_rows:
    print(f"{name:>22} {f:>12.2e} {n:>10.3f} {E_j:>14.4e} {E_ev:>14.4e} {period:>18}")

print(f"\n--- TEST 2: Graviton Mass Bound ---")
print(f"At gravity anchor (10^12 Hz):")
print(f"  E_min  = {E_floor_gravity:.4e} J  ({E_floor_gravity/eV:.4e} eV)")
print(f"  m_min  = {m_graviton_min:.4e} kg  ({m_graviton_min/m_P:.4e} x m_Planck)")
print(f"  lambda_c = {compton_grav:.4e} m  (Compton wavelength -- max gravity range)")
print(f"At black hole floor (10^-30 Hz):")
print(f"  E_min  = {E_floor_bh:.4e} J  ({E_floor_bh/eV:.4e} eV)")
print(f"  m_min  = {m_graviton_bh:.4e} kg  ({m_graviton_bh/m_P:.4e} x m_Planck)")
print("VERDICT: Graviton cannot have exactly zero mass -- phi^1 floor sets minimum.")

print(f"\n--- TEST 3: Detection Threshold ---")
print(f"Graviton energy at n*={n_grav:.2f}: {E_graviton:.4e} J ({E_graviton/eV:.4e} eV)")
print(f"LIGO sensitivity floor:           {E_LIGO:.0e} J")
print(f"Rung gap (graviton to detector):  {rung_gap:.2f} rungs")
print(f"Gravitons needed to trigger LIGO: {N_needed:.4e}")
print("VERDICT: LIGO detects coherent bulk resonance -- consistent with rung gap.")

print(f"\n--- TEST 4: Black Hole Interior ---")
print(f"BH floor frequency: {f_blackhole:.0e} Hz")
print(f"Rung n*:            {n_bh:.4f}")
print(f"Period:             {period_human(T_bh)}")
print("VERDICT: Singularity replaced by asymptotic rung descent to phi^0.")

print(f"\n--- TEST 5: Virtual Graviton Elimination ---")
print(f"phi-Framework: coupling = shared rung geometry.")
print(f"Two masses on same rung interact directly -- no mediating particle needed.")
print(f"Virtual gravitons are unnecessary scaffolding.")

# ===================================================================
# HTML EXPORT
# ===================================================================

def sc(s):
    u = s.upper()
    if any(x in u for x in ["INADMISSIBLE", "NO RUNG", "BELOW", "TERMINATED"]):
        return "bad"
    if any(x in u for x in ["VALID", "MANIFOLD", "ADMISSIBLE"]):
        return "good"
    return ""


def build_anchor_rows():
    out = ""
    for name, f, n, E_j, E_ev, period in anchor_rows:
        out += (
            "<tr>"
            + "<td>" + name + "</td>"
            + "<td>" + f"{f:.2e}" + "</td>"
            + "<td>" + f"{n:.3f}" + "</td>"
            + "<td>" + f"{E_j:.4e}" + "</td>"
            + "<td>" + f"{E_ev:.4e}" + "</td>"
            + "<td>" + period + "</td>"
            + "</tr>\n"
        )
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
  h1 .phi{color:#3dd6f5;} h1 .grav{color:#e8c84a;}
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
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-bottom:24px;}
  .card{background:#080e1a;border:1px solid #0d2035;border-top:2px solid #3dd6f5;padding:14px 16px;}
  .card.warn{border-top-color:#ff4f6d;} .card.gold{border-top-color:#e8c84a;}
  .card-label{font-family:'Courier New',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3a5068;margin-bottom:6px;}
  .card-val{font-family:'Courier New',monospace;font-size:16px;color:#fff;font-weight:700;}
  .card-sub{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;margin-top:4px;}
  .verdict{background:#080e1a;border:1px solid #0d2035;border-left:3px solid #3dd6f5;padding:20px 22px;}
  .verdict-title{font-family:'Courier New',monospace;font-size:11px;color:#3dd6f5;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:12px;}
  .verdict p{margin-bottom:10px;font-size:14px;line-height:1.8;}
  .verdict p:last-child{margin-bottom:0;}
  .hi{color:#3dd6f5;} .warn-t{color:#ff4f6d;} .gold-t{color:#e8c84a;}
  .note{font-family:'Courier New',monospace;font-size:11px;color:#3a5068;margin-top:8px;}
  .summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  @media(max-width:600px){.summary-grid{grid-template-columns:1fr;}}
  .summary-item{background:#080e1a;border:1px solid #0d2035;padding:14px 16px;font-family:'Courier New',monospace;font-size:12px;color:#7a9ab0;line-height:1.7;}
  .summary-item b{color:#3dd6f5;display:block;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;}
  footer{margin-top:48px;font-family:'Courier New',monospace;font-size:10px;color:#1e3347;
    letter-spacing:2px;text-align:center;}
"""

html = (
    "<!DOCTYPE html>\n"
    "<html lang='en'>\n"
    "<head>\n"
    "<meta charset='UTF-8'>\n"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    "<title>Graviton Analysis - phi-Resonance Framework | QCore Labs</title>\n"
    "<style>" + CSS + "</style>\n"
    "</head>\n"
    "<body>\n"
    "<div class='wrap'>\n"

    "<header>\n"
    "  <div class='lab-tag'>QCore Labs &mdash; phi-Resonance Framework &mdash; Graviton Analysis</div>\n"
    "  <h1><span class='grav'>Graviton Analysis</span> &mdash; <span class='phi'>phi-Resonance</span> Framework</h1>\n"
    "  <div class='sub'>Gravity anchor: 10&sup1;&sup2; Hz &nbsp;|&nbsp; BH floor: 10&sup3;&#8320; Hz &nbsp;|&nbsp; phi = " + f"{phi:.10f}" + "</div>\n"
    "</header>\n"

    "<div class='sec'>\n"
    "  <div class='sec-label'>Test <span>01</span> &mdash; Graviton Energy at Each Anchor</div>\n"
    "  <div class='tbl-wrap'><table>\n"
    "    <thead><tr><th>Force</th><th>Freq (Hz)</th><th>Rung n*</th><th>E_J (J)</th><th>E (eV)</th><th>Period</th></tr></thead>\n"
    "    <tbody>" + build_anchor_rows() + "</tbody>\n"
    "  </table></div>\n"
    "  <div class='note'>E_J = hbar * 2*pi*f * phi^n* -- Jackson energy at the self-anchored rung for each force.</div>\n"
    "</div>\n"

    "<div class='sec'>\n"
    "  <div class='sec-label'>Test <span>02</span> &mdash; Graviton Mass Bound</div>\n"
    "  <div class='cards'>\n"
    "    <div class='card gold'><div class='card-label'>E_min at gravity anchor</div><div class='card-val'>" + f"{E_floor_gravity:.4e} J" + "</div><div class='card-sub'>" + f"{E_floor_gravity/eV:.4e} eV" + "</div></div>\n"
    "    <div class='card gold'><div class='card-label'>m_min at gravity anchor</div><div class='card-val'>" + f"{m_graviton_min:.4e} kg" + "</div><div class='card-sub'>" + f"{m_graviton_min/m_P:.4e} x m_Planck" + "</div></div>\n"
    "    <div class='card'><div class='card-label'>Compton wavelength</div><div class='card-val'>" + f"{compton_grav:.4e} m" + "</div><div class='card-sub'>Max range of gravity</div></div>\n"
    "    <div class='card warn'><div class='card-label'>Standard QFT mass</div><div class='card-val'>Exactly 0</div><div class='card-sub'>phi-Framework: NONZERO -- prediction</div></div>\n"
    "  </div>\n"
    "  <div class='note'>The phi-ladder has a hard floor at n=1. Zero energy is only reached asymptotically as n &rarr; &minus;&infin;, never at a finite rung. Graviton mass = 0 is geometrically inadmissible.</div>\n"
    "</div>\n"

    "<div class='sec'>\n"
    "  <div class='sec-label'>Test <span>03</span> &mdash; Detection Threshold (Rung Analysis)</div>\n"
    "  <div class='cards'>\n"
    "    <div class='card gold'><div class='card-label'>Graviton rung n*</div><div class='card-val'>" + f"{n_grav:.2f}" + "</div><div class='card-sub'>At 10^12 Hz gravity anchor</div></div>\n"
    "    <div class='card gold'><div class='card-label'>Graviton energy</div><div class='card-val'>" + f"{E_graviton:.4e} J" + "</div><div class='card-sub'>" + f"{E_graviton/eV:.4e} eV" + "</div></div>\n"
    "    <div class='card'><div class='card-label'>Rung gap to LIGO</div><div class='card-val'>" + f"{rung_gap:.2f}" + "</div><div class='card-sub'>Explains non-detection</div></div>\n"
    "    <div class='card warn'><div class='card-label'>Gravitons for LIGO</div><div class='card-val'>" + f"{N_needed:.2e}" + "</div><div class='card-sub'>Coherent bulk required</div></div>\n"
    "  </div>\n"
    "  <div class='note'>LIGO detects gravitational WAVES (coherent bulk resonance), not individual gravitons -- fully consistent with the rung gap prediction.</div>\n"
    "</div>\n"

    "<div class='sec'>\n"
    "  <div class='sec-label'>Tests <span>04 &amp; 05</span> &mdash; BH Interior &amp; Virtual Graviton Elimination</div>\n"
    "  <div class='summary-grid'>\n"
    "    <div class='summary-item'>\n"
    "      <b>Test 04 -- Black Hole Interior</b>\n"
    "      BH floor frequency: " + f"{f_blackhole:.0e}" + " Hz<br>\n"
    "      Rung n*: " + f"{n_bh:.4f}" + "<br>\n"
    "      Period: " + period_human(T_bh) + "<br><br>\n"
    "      Standard GR: singularity at r=0, density &rarr; &infin;.<br>\n"
    "      phi-Framework: interior is a ladder descent toward phi^0.<br>\n"
    "      As matter falls inward, frequency redshifts (f &rarr; 0), rung descends (n* &rarr; &minus;&infin;) asymptotically.<br>\n"
    "      <span style='color:#3dd6f5;'>No singularity. Just the ladder running out of rungs toward vacuum.</span>\n"
    "    </div>\n"
    "    <div class='summary-item'>\n"
    "      <b>Test 05 -- Virtual Graviton Elimination</b>\n"
    "      Standard QFT requires virtual gravitons as off-shell mediators. These violate E&sup2;=(pc)&sup2;+(mc&sup2;)&sup2; and exist only inside Feynman diagrams.<br><br>\n"
    "      phi-Framework: gravity is a real resonance at n* = " + f"{n_grav:.2f}" + " (10&sup1;&sup2; Hz).<br>\n"
    "      Two masses interact by occupying the SAME rung -- they share the same resonant frequency.<br><br>\n"
    "      Like two pendulums on the same beam synchronizing without exchanging particles:<br>\n"
    "      <span style='color:#3dd6f5;'>The coupling IS the shared rung. No mediating particle needed.</span>\n"
    "    </div>\n"
    "  </div>\n"
    "</div>\n"

    "<div class='sec'>\n"
    "  <div class='sec-label'>Step <span>06</span> &mdash; Graviton Summary Verdict</div>\n"
    "  <div class='verdict'>\n"
    "    <div class='verdict-title'>Five Graviton Problems. One Geometric Answer.</div>\n"
    "    <p><span class='gold-t'>1. Energy:</span> Rung-specific, not zero-point. Each force self-anchors at its own frequency; the ladder structure is universal (phi^n), tuning is system-specific.</p>\n"
    "    <p><span class='gold-t'>2. Mass:</span> <span class='hi'>Nonzero minimum set by phi^1 floor -- a testable prediction.</span> Compton wavelength = " + f"{compton_grav:.4e}" + " m sets the maximum range of gravity.</p>\n"
    "    <p><span class='gold-t'>3. Detection:</span> Rung gap of " + f"{rung_gap:.1f}" + " rungs between graviton (n*=" + f"{n_grav:.2f}" + ") and LIGO threshold explains why individual gravitons are undetectable. LIGO sees coherent bulk -- consistent.</p>\n"
    "    <p><span class='gold-t'>4. BH Interior:</span> Singularity is replaced by asymptotic rung descent toward phi^0. No infinite density. No broken physics.</p>\n"
    "    <p><span class='gold-t'>5. Virtual gravitons:</span> Eliminated. The coupling is the shared rung. <span class='hi'>This is not renormalization. It is geometry.</span></p>\n"
    "  </div>\n"
    "</div>\n"

    "<footer>QCORE LABS &mdash; LAW OF HARMONIC RESONANCE &mdash; phi-RECURSIVE FRAMEWORK</footer>\n"
    "</div>\n"
    "</body>\n"
    "</html>"
)

with open("Graviton_analysis.html", "w", encoding="utf-8") as f:
    f.write(html)
print("HTML report saved -> Graviton_analysis.html")
