"""Reproduce the Application: coverage of all six regions and the two figures.

Run from the repository root (with the ``paper`` extra installed for matplotlib)::

    python -m examples.bilinear_family --outdir .

This prints the coverage diagnostics and writes ``phase_diagram_a05_c1.png`` and
``selection_confirmation_a05_c1.png`` -- the two figures of the Application section.
The numerics use only NumPy; matplotlib is required solely for the figures.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from o2sym import (
    ConservativeChecker,
    application_1_system,
    extract_quadratic_forms,
    hopf_params,
    minimal_subsets,
    normal_form_coeffs,
    reachable_regions,
    scan_grid,
    simulate_system2,
)
from o2sym.coverage import bilinear_nonlinearity

A_DEFAULT, C_DEFAULT, G21 = 0.5, 1.0, 1.0
PT_IV: Tuple[float, float, float] = (0.40, 1.0, 0.30)   # standing wave
PT_V: Tuple[float, float, float] = (0.35, 1.0, -0.30)   # traveling wave
_REGION_STABILITY = {
    "I": "TW saddle, SW unstable", "II": "TW unstable, SW saddle",
    "III": "TW saddle (no SW)", "IV": "TW saddle, SW stable",
    "V": "TW stable, SW saddle", "VI": "SW saddle (no TW)",
}


def print_coverage(a: float, c: float) -> None:
    """Print conservativeness, the quadratic forms, and the coverage / minimality results."""
    print(f"\n=== Coverage at a={a}, c={c} (Li--Yao operator, m_c=1, delta={1 + a}) ===")
    nl = bilinear_nonlinearity(1.0, 1.0, 1.0)
    for comp in (0, 1):
        chk = ConservativeChecker.from_nonlinearity_component(nl, comp)
        print(f"  component {comp + 1} conservative: {chk.is_conservative()}")
    beta = application_1_system().diagnostic_data(1, hopf_params(a, c))["beta_mc"]
    print(f"  beta at m_c       : {beta[0]:.6g}, {beta[1]:.6g}")
    A, B = extract_quadratic_forms(a, c)
    np.set_printoptions(precision=6, suppress=True)
    print(f"  A (Re zeta = g^T A g):\n{A}")
    print(f"  B (Re xi   = g^T B g):\n{B}")
    rr = reachable_regions(a, c)
    print(f"  regions reachable by the full family: {sorted(rr['regions'])} "
          f"(count {rr['count']}, angular span {rr['angular_coverage_deg']:.1f} deg)")
    ms = minimal_subsets(a, c)
    print(f"  subsets reaching all six            : {ms['reach_all_six']}")
    print(f"  with g11 = 0 (subset g21,g22)       : {sorted(ms[('g21', 'g22')])}")


def make_phase_diagram(a: float, c: float, path: str, half_width: float = 0.5, n: int = 1400) -> None:
    """Write the (g11, g22)-plane wave-selection phase diagram at fixed g21 = 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    sc = scan_grid(a, c, g21=G21, half_width=half_width, n=n)
    G11, G22, Z, X, R = sc.g11, sc.g22, sc.re_zeta, sc.re_xi, sc.region
    cols = ["#f4a6a6", "#f6d186", "#cfe8a9", "#a9d6e8", "#b9a9e8", "#e8a9d4"]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.pcolormesh(G11, G22, R, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.contour(G11, G22, Z, levels=[0], colors="k", linewidths=1.6)
    ax.contour(G11, G22, Z + X, levels=[0], colors="k", linewidths=1.4, linestyles="--")
    ax.contour(G11, G22, Z - X, levels=[0], colors="k", linewidths=1.4, linestyles=":")
    ax.plot(0, 0, marker="*", ms=18, mfc="white", mec="k", mew=1.3, zorder=5)
    ax.annotate("Li--Yao\n$(g_{11},g_{22})=(0,0)$", (0, 0), textcoords="offset points",
                xytext=(10, 8), fontsize=9, zorder=6)
    names = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    for k in range(1, 7):
        mask = R == k
        if mask.sum() > 50:
            ax.text(G11[mask].mean(), G22[mask].mean(), names[k], ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="k", alpha=0.85))
    ax.set_xlabel(r"$g_{11}$  (coefficient of $u_xv+uv_x$)", fontsize=11)
    ax.set_ylabel(r"$g_{22}$  (coefficient of $v\,v_x$)", fontsize=11)
    ax.set_title(r"Wave-selection regions of the conservative bilinear family"
                 "\n" r"$a=\frac{1}{2},\ c=1$ (Li--Yao operator, $m_c=1$), $g_{21}=1$",
                 fontsize=11)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    leg_regions = [Patch(fc=cols[i], ec="k", label=f"{names[i + 1]}: {_REGION_STABILITY[names[i + 1]]}")
                   for i in range(6)]
    leg_lines = [Line2D([0], [0], color="k", lw=1.6, label=r"$\mathrm{Re}\,\zeta=0$"),
                 Line2D([0], [0], color="k", lw=1.4, ls="--", label=r"$\mathrm{Re}(\zeta+\xi)=0$"),
                 Line2D([0], [0], color="k", lw=1.4, ls=":", label=r"$\mathrm{Re}(\zeta-\xi)=0$")]
    l1 = ax.legend(handles=leg_regions, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   fontsize=8.5, title="region (stability)", title_fontsize=9, frameon=True)
    ax.add_artist(l1)
    ax.legend(handles=leg_lines, loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (regions present: {sorted(sc.regions_present)})")


def _critical_growth_and_mode(a: float, c: float, mu: float) -> Tuple[float, float]:
    """Return (Re beta at m_c, |q_1^{(u)}|) at delta = (1+a)+mu, using the growing eigenpair."""
    params = dict(hopf_params(a, c)); params["delta"] = (1.0 + a) + mu
    d = application_1_system().diagnostic_data(1, params)
    i = 0 if d["beta_mc"][0].real >= d["beta_mc"][1].real else 1
    beta_r = d["beta_mc"][i].real
    q1 = abs(d["q_mc"][i][0])   # modulus of the u-component of the critical eigenvector
    return beta_r, q1


def make_confirmation(a: float, c: float, path: str,
                      mus=(0.025, 0.05, 0.1, 0.15)) -> None:
    """Write the PDE confirmation figure (space-time, |u_+1| pulsing, amplitude scaling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 128
    runs = {"IV": (PT_IV, "standing wave"), "V": (PT_V, "traveling wave")}
    data = {}
    for tag, (g, _) in runs.items():
        r = simulate_system2(a, c, g, delta_minus_dc=0.05, N=N, T=1000.0, tail=45.0, seed=1)
        data[tag] = r
        print(f"  Region {tag}: g={g}  modulation depth {r.modulation_depth:.2f} -> {r.label}")

    # parameter-free cubic-theory line for the traveling-wave (Region V) point:
    # |u_hat_+1| = N |q_1| sqrt( Re beta(mu) / |Re zeta| )
    rez_v = float(np.real(normal_form_coeffs(PT_V, a, c)[0]))
    amps = []
    for mu in mus:
        rr = simulate_system2(a, c, PT_V, delta_minus_dc=mu, N=N, T=80.0 / mu, tail=2.0, seed=1)
        amps.append(rr.saturated_amplitude)
    pred = []
    for mu in mus:
        beta_r, q1 = _critical_growth_and_mode(a, c, mu)
        pred.append(N * q1 * np.sqrt(max(beta_r, 0.0) / abs(rez_v)))
    slope = float(np.polyfit(np.log(mus), np.log(amps), 1)[0])

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.1], hspace=0.42, wspace=0.42)
    for i, (tag, (g, desc)) in enumerate(runs.items()):
        r = data[tag]
        t0 = r.snap_times[0]
        ax = fig.add_subplot(gs[i, 0])
        vmax = np.abs(r.snapshots).max()
        ax.pcolormesh(r.x, r.snap_times - t0, r.snapshots, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(f"Region {tag}: $u(x,t)$  $\\to$ {desc}", fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$ (last 45 units)")
        ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
        ax2 = fig.add_subplot(gs[i, 1])
        amp = r.mode1_amplitude
        ax2.plot(r.snap_times - t0, amp / amp.max(), "k", lw=1.2)
        ax2.set_ylim(-0.05, 1.1); ax2.set_xlabel("$t$")
        ax2.set_ylabel("$|\\hat u_{+1}(t)|$ (norm.)")
        ax2.set_title(f"modulation depth = {r.modulation_depth:.2f}", fontsize=10)
    axs = fig.add_subplot(gs[:, 2])
    (l_thy,) = axs.plot(np.sqrt(mus), pred, "-", color="tab:blue",
                        label=r"cubic theory $\propto(\delta-\delta_c)^{1/2}$")
    (l_sim,) = axs.plot(np.sqrt(mus), amps, linestyle="None", marker="o", color="k", ms=7,
                        label="PDE simulation (Region V)")
    axs.set_xlabel(r"$\sqrt{\delta-\delta_c}$")
    axs.set_ylabel(r"saturated $|\hat u_{+1}|$")
    axs.set_title(f"amplitude scaling (log-log slope {slope:.2f})", fontsize=10)
    axs.legend(handles=[l_thy, l_sim], fontsize=9, loc="lower right")
    axs.set_xlim(0, 0.43); axs.set_ylim(0, None)
    fig.suptitle(r"PDE confirmation of wave selection  "
                 r"($a=\frac{1}{2},\,c=1$, Li--Yao operator, $g_{21}=1$)", fontsize=12)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (Re zeta_V={rez_v:+.4f}, log-log amplitude slope={slope:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the Application coverage and figures.")
    ap.add_argument("--outdir", default=".", help="directory for the output PNGs")
    ap.add_argument("--a", type=float, default=A_DEFAULT)
    ap.add_argument("--c", type=float, default=C_DEFAULT)
    ap.add_argument("--no-figures", action="store_true", help="print coverage only")
    args = ap.parse_args()

    print_coverage(args.a, args.c)
    if not args.no_figures:
        os.makedirs(args.outdir, exist_ok=True)
        make_phase_diagram(args.a, args.c, os.path.join(args.outdir, "phase_diagram_a05_c1.png"))
        make_confirmation(args.a, args.c, os.path.join(args.outdir, "selection_confirmation_a05_c1.png"))


if __name__ == "__main__":
    main()
