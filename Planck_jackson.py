"""
Planck vs Jackson Ladder Divergence - phi-Resonance Framework
QCore Labs -- High-Precision Analysis

Replicates all six steps from the HTML report:
1. Core ladder comparison (n = 1..12)
2. Second finite differences of Delta_n
3. Exponential fit and curvature mismatch
4. Planck-5c projection onto Jackson manifold
5. Wormhole / exotic matter off-manifold analysis
6. Analytic verdict + matplotlib divergence chart
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar, curve_fit

# -- Constants -----------------------------------------------------------------

phi = (1 + np.sqrt(5)) / 2   # 1.6180339887498948...

print("=" * 70)
print("PLANCK vs JACKSON LADDER DIVERGENCE")
print("QCore Labs -- phi-Resonance Framework")
print("=" * 70)
print(f"\nphi  = {phi:.16f}")
print(f"phi^2 = {phi**2:.16f}")


# ==============================================================================
# STEP 1 -- Core Ladder Comparison
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 01 -- Core Ladder Comparison  (n = 1 to 12)")
print("=" * 70)

ns = list(range(1, 13))
EP = [float(n) for n in ns]
EJ = [phi**n for n in ns]
Delta = [ej - ep for ej, ep in zip(EJ, EP)]
Ratio = [ej / ep for ej, ep in zip(EJ, EP)]

print(f"\n{'n':>4}  {'E_P = n':>14}  {'E_J = phi^n':>16}  {'Delta':>16}  {'R = E_J/E_P':>14}")
print("-" * 70)
for i, n in enumerate(ns):
    marker = " *" if n == 5 else "  "
    print(f"{n:>4}{marker} {EP[i]:>14.10f}  {EJ[i]:>16.10f}  {Delta[i]:>+16.10f}  {Ratio[i]:>14.10f}")

print("\n  * = Planck-5c highlighted")


# ==============================================================================
# STEP 2 -- Second Finite Differences of Delta_n
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 02 -- Second Finite Differences of Delta_n")
print("=" * 70)

fd2 = []
fd2_ns = []
for i in range(1, len(Delta) - 1):
    d2 = Delta[i+1] - 2*Delta[i] + Delta[i-1]
    fd2.append(d2)
    fd2_ns.append(ns[i])

print(f"\n{'n':>4}  {'Delta^2 Delta_n':>18}  {'Sign':>6}  {'Monotone?':>10}")
print("-" * 44)
for i, n in enumerate(fd2_ns):
    sign = "+" if fd2[i] > 0 else "-"
    mono = "YES" if fd2[i] > 0 else "NO"
    print(f"{n:>4}  {fd2[i]:>18.10f}  {sign:>6}  {mono:>10}")

first_mono = fd2_ns[0] if all(v > 0 for v in fd2) else "N/A"
print(f"\n-> Monotone from n = {first_mono}. Ladders never tangent after n = 1.")


# ==============================================================================
# STEP 3 -- Exponential Fit and Curvature Mismatch
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 03 -- Exponential Fit and Curvature Mismatch")
print("=" * 70)

# Jackson: exact fit E_n = 1 * phi^n
a_J = 1.0
b_J = phi
print(f"\nJackson fit  E_n = a * phi^n:")
print(f"  a = {a_J:.8f}")
print(f"  b = {b_J:.8f}  (= phi, exact)")
print(f"  b_J / phi = {b_J/phi:.12f}  (check)")

# Planck: force fit E_n = a * b^n via log-linear regression
log_EP = np.log(EP)
log_ns = np.array(ns, dtype=float)
coeffs = np.polyfit(log_ns, log_EP, 1)   # log(E) = b*log(n) + log(a) -- but E_P=n so linear
# Better: fit a*b^n directly
def exp_model(n, a, b):
    return a * b**n

popt, _ = curve_fit(exp_model, ns, EP, p0=[0.5, 1.2])
a_P, b_P = popt

print(f"\nPlanck forced fit  E_n = a * b_P^n:")
print(f"  a   = {a_P:.8f}")
print(f"  b_P = {b_P:.8f}")
print(f"  |phi - b_P|   = {abs(phi - b_P):.8f}")
print(f"  phi / b_P     = {phi/b_P:.6f}")
print(f"\nPlanck effective base is {(1 - b_P/phi)*100:.1f}% below phi -- structural incommensurability.")


# ==============================================================================
# STEP 4 -- Planck-5c Projection onto Jackson Manifold
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 04 -- Planck-5c Projection onto Jackson Manifold")
print("=" * 70)

# Planck-5c point: (n=5, E=5)
p5_n = 5.0
p5_E = 5.0

# Jackson manifold: parametric curve (t, phi^t) for t in [1, 12]
# Find t* that minimizes distance from (5, 5) to (t, phi^t)
def dist_sq(t):
    return (t - p5_n)**2 + (phi**t - p5_E)**2

result = minimize_scalar(dist_sq, bounds=(1.0, 12.0), method="bounded")
t_star = result.x
E_star = phi**t_star
d_perp = np.sqrt(dist_sq(t_star))

nearest_rung = round(t_star)
delta_n = t_star - p5_n

print(f"\nPlanck-5c point:          (n=5, E=5.0000)")
print(f"Closest Jackson point:    (t* = {t_star:.8f}, E = {E_star:.6f})")
print(f"Orthogonal distance d_perp = {d_perp:.4f}")
print(f"  As % of E_5^J = {phi**5:.6f}:  {d_perp/phi**5*100:.2f}%")
print(f"Delta_n from nearest rung: {delta_n:+.4f}")
print(f"Nearest integer rung:      n = {nearest_rung}")


# ==============================================================================
# STEP 5 -- Wormhole / Exotic Matter Off-Manifold
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 05 -- Wormhole / Exotic Matter Off-Manifold")
print("=" * 70)

# Morris-Thorne: throat radius r0 = 5 Planck lengths
r0 = 5.0
n_r0 = np.log(r0) / np.log(phi)   # Jackson rung for r0

# Exotic matter density (natural units): rho = -c^4 / (8*pi*G*r0^2)
# In Planck units: rho_exotic = -1 / (8*pi*r0^2)
rho_exotic = -1.0 / (8 * np.pi * r0**2)

# Attempt to map rho to rung
if rho_exotic > 0:
    n_rho = np.log(rho_exotic) / np.log(phi)
else:
    n_rho = None   # negative -- no valid rung

floor_energy = phi**1   # Jackson ground state n=1

off_manifold_delta = abs(r0 - phi**3)   # |5 - phi^3|

print(f"\nMorris-Thorne throat radius:  r0 = {r0:.4f} Planck lengths")
print(f"Jackson rung for r0 = 5:      n* = {n_r0:.4f}  (non-integer -- no exact rung)")
print(f"Off-manifold delta in E:       |r0 - phi^3| = |{r0} - {phi**3:.4f}| = {off_manifold_delta:.4f}")
print(f"\nExotic matter density (Planck units): rho = {rho_exotic:.6f}")
print(f"Jackson ground state floor:           phi^1 = {floor_energy:.6f}")
if n_rho is None:
    print(f"Exotic rho rung:  NEGATIVE -- no valid Jackson rung exists")
    print(f"  -> Exotic matter is entirely off-ladder (below n=1 floor)")
else:
    print(f"Exotic rho rung:  n* = {n_rho:.4f}")

# Compute n* for rho if we use magnitude to show how far below floor
n_rho_mag = np.log(abs(rho_exotic)) / np.log(phi)
print(f"  Magnitude rung (for reference): n* = {n_rho_mag:.4f}  (BELOW n=1 floor -- inadmissible)")


# ==============================================================================
# STEP 6 -- Analytic Verdict
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 06 -- Analytic Verdict")
print("=" * 70)
print(f"""
PLANCK CONSTRUCTIONS ARE OFF-MANIFOLD
--------------------------------------
The second finite difference of Delta_n is strictly positive and
monotone from n = {first_mono}. The linear Planck axis is never tangent
to the phi-recursive Jackson manifold after the first rung.

Curvature mismatch: fitted Planck base b_P = {b_P:.4f} vs phi = {phi:.4f}.
Gap = {abs(phi-b_P):.4f}, ratio phi/b_P = {phi/b_P:.4f}. Structural, not perturbative.

Planck-5c projects at t* = {t_star:.3f} with orthogonal displacement
d_perp = {d_perp:.4f} ({d_perp/phi**5*100:.1f}% of E_5^J).
Nearest integer rung is n = {nearest_rung}, not n = 5.

Morris-Thorne wormhole throat r0 = 5 Planck lengths lands at
Jackson n* = {n_r0:.3f} (non-integer, delta = {off_manifold_delta:.3f}).
Required exotic matter density maps to n* = {n_rho_mag:.2f} -- below
the Jackson ground state. No valid eigenstate exists there.

Conclusion: particles and states derived using the Planck ladder
as scaling base have no corresponding Jackson rung. They are
solutions to a misspecified eigenvalue equation -- artifacts of a
linear grid imposed on a phi-recursive vacuum.
""")


# ==============================================================================
# MATPLOTLIB CHART -- Ladder Divergence
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#04080f")

ns_arr = np.array(ns)
EJ_arr = np.array(EJ)
EP_arr = np.array(EP)

# ---- LEFT: Ladder comparison -------------------------------------------------
ax1 = axes[0]
ax1.set_facecolor("#080e1a")
ax1.grid(color="#0d2035", linewidth=0.8, zorder=0)

ax1.plot(ns_arr, EJ_arr, color="#3dd6f5", linewidth=2.5,
         marker="o", markersize=5, label="Jackson $E_n = \\phi^n$", zorder=3)
ax1.plot(ns_arr, EP_arr, color="#e8c84a", linewidth=2.0,
         linestyle="--", marker="o", markersize=4,
         label="Planck $E_n = n$", zorder=3)

# Planck-5c highlight
ax1.scatter([5], [5], color="#ff4f6d", s=120, zorder=5)
ax1.annotate("Planck-5c\n$d_\\perp$ = 1.540",
             xy=(5, 5), xytext=(6.2, 12),
             color="#ff4f6d", fontsize=9, fontfamily="monospace",
             arrowprops=dict(arrowstyle="->", color="#ff4f6d", lw=1.2))

# Projection line to Jackson manifold
ax1.plot([5, t_star], [5, E_star],
         color="#ff4f6d", linewidth=1.5, linestyle=":", zorder=4)

ax1.set_xlim(0.5, 12.5)
ax1.set_ylim(-5, 135)
ax1.set_xlabel("n", color="#3a5068", fontsize=11)
ax1.set_ylabel("Energy (a.u.)", color="#3a5068", fontsize=11)
ax1.set_title("Ladder Divergence", color="#c8dde8", fontsize=12, pad=10)
ax1.tick_params(colors="#3a5068")
for spine in ax1.spines.values():
    spine.set_edgecolor("#0d2035")
ax1.legend(fontsize=9, facecolor="#080e1a", edgecolor="#0d2035",
           labelcolor="#c8dde8")

# ---- RIGHT: Residual Delta_n bar chart ---------------------------------------
ax2 = axes[1]
ax2.set_facecolor("#080e1a")
ax2.grid(color="#0d2035", linewidth=0.8, zorder=0, axis="y")

colors_bar = ["#ff4f6d" if d > 0 else "#4a6a80" for d in Delta]
ax2.bar(ns_arr, Delta, color=colors_bar, edgecolor="#0d2035",
        linewidth=0.8, zorder=3)

ax2.axhline(0, color="#3a5068", linewidth=0.8, linestyle="--")

# Annotate monotone onset
ax2.annotate(f"Monotone from n={first_mono}",
             xy=(first_mono, Delta[first_mono-1]),
             xytext=(first_mono + 1.5, Delta[first_mono-1] * 0.6),
             color="#3dd6f5", fontsize=9, fontfamily="monospace",
             arrowprops=dict(arrowstyle="->", color="#3dd6f5", lw=1.0))

ax2.set_xlim(0.5, 12.5)
ax2.set_xlabel("n", color="#3a5068", fontsize=11)
ax2.set_ylabel("Delta_n = E_J - E_P", color="#3a5068", fontsize=11)
ax2.set_title("Residual per Rung", color="#c8dde8", fontsize=12, pad=10)
ax2.tick_params(colors="#3a5068")
for spine in ax2.spines.values():
    spine.set_edgecolor("#0d2035")

plt.suptitle("Planck vs Jackson Ladder  |  QCore Labs",
             color="#c8dde8", fontsize=13, y=1.01)
plt.tight_layout()

outpath = "Planck_Jackson_divergence.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight",
            facecolor="#04080f")
print(f"Chart saved -> {outpath}")
plt.close()
