"""Architecture diagram of the COSMOS-AI digital twin (vector PDF + 300 dpi PNG).

    python paper/journal_v3_twin/tools/make_architecture.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artwork")
INK, INK2, GREY = "#0b0b0b", "#52514e", "#ecebe6"
H_C, D_C, R_C, S_C = "#dbe8f8", "#fde3d6", "#e3f1df", "#f3eefc"


def box(ax, x, y, w, h, fc, title, body, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=fc,
                                edgecolor=INK2, linewidth=0.8))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=fs + 1, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.3)


def arrow(ax, a, b, text="", dashed=False, rad=0.0, tx=0, ty=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=1.0,
                                 linestyle=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, fontsize=6.4, color=INK2, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def main():
    fig = plt.figure(figsize=(7.4, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.80, 0.96, 0.17, GREY, "Data (real)",
        "GCAT catalogue (64,516 LEO objects)  ·  CelesTrak SATCAT, operational status, RCS, space weather\n"
        "ESA Kelvins conjunction messages (13,154 + 2,167 events)  ·  ESA-ADB and NASA SMAP/MSL telemetry")
    box(ax, 0.30, 0.50, 0.40, 0.22, S_C, "Object twin  s_i(t)",
        "orbit · design properties · health status\n"
        "debris family (causal) · space weather\nconjunction summary · module outputs + provenance")
    box(ax, 0.02, 0.06, 0.29, 0.34, H_C, "H  Satellite health",
        "forecaster + conformal p-values\nBH across channels, persistence\n(ESA-ADB companion\nstudy; SMAP/MSL)\n\nout: alarms, operational status")
    box(ax, 0.355, 0.06, 0.29, 0.34, D_C, "D  Orbital debris",
        "breakup / launch family evidence\nconjunction risk: persistence +\nlearned experts, gate, conformal\n(ESA Kelvins CAC)\n\nout: P(high risk), risk interval,\ncloud survival")
    box(ax, 0.69, 0.06, 0.29, 0.34, R_C, "R  Re-entry risk",
        "physics prior: King-Hele drag,\nNRLMSIS 2.1, breakup-model A/m (MC)\nlearned expert: TabM, TabICLv2,\nKumo-Tabular (trees as baselines)\nreliability gate + Mondrian conformal\n\nout: time-to-re-entry distribution,\nP(re-entry within H)")
    for x in (0.165, 0.50, 0.835):
        arrow(ax, (x, 0.40), (0.50 + (x - 0.5) * 0.35, 0.50))
    arrow(ax, (0.5, 0.80), (0.5, 0.72))
    arrow(ax, (0.645, 0.20), (0.69, 0.20), "C1 family", ty=0.03)
    arrow(ax, (0.69, 0.13), (0.645, 0.13), "C2 cloud survival", ty=-0.03)
    arrow(ax, (0.24, 0.40), (0.76, 0.40), "C3 status → regime", rad=-0.18, tx=0.13, ty=0.045)
    arrow(ax, (0.31, 0.24), (0.355, 0.24), "C4", dashed=True, ty=0.025)
    ax.text(0.5, 0.012, "Solid: couplings tested on real data.  Dashed: demonstration only (no public paired data).  "
            "One uncertainty principle: conformal calibration in every module.", ha="center", fontsize=6.4, color=INK2)
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_figures import fig_name
    name = fig_name("architecture")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white", **kw)
    print("wrote", name)


if __name__ == "__main__":
    main()
