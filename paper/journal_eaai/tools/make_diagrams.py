"""Schematic diagrams for the Engineering Applications of Artificial Intelligence manuscript.

    python paper/journal_eaai/tools/make_diagrams.py

Two diagrams, both describing the implemented pipeline and nothing else:

* ``eaai_method``      the experimental flow of the re-entry module, from catalogue to scored test
                       cohort, with the split boundaries and the stage on which each component is
                       fitted (``cosmos/ai/lifetime/run.py``);
* ``eaai_integration`` the four cross-module couplings as a data flow, with the variable that
                       crosses each boundary (``cosmos/ai/twin/couplings.py``, ``engine.py``).

No quantity is invented: every label is a stage, a variable or a split rule that exists in the code.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artwork_eaai")
INK, INK2, GREY = "#0b0b0b", "#52514e", "#ecebe6"
DATA, PHYS, LEARN, FUSE, EVAL = "#ecebe6", "#f3eefc", "#e3f1df", "#fde3d6", "#dbe8f8"


def box(ax, x, y, w, h, fc, title, body="", fs=6.6, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                facecolor=fc, edgecolor=INK2, linewidth=0.8))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=fs + 0.8,
            fontweight="bold" if bold else "normal", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.058, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, a, b, text="", dashed=False, rad=0.0, tx=0.0, ty=0.0, fs=6.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=INK2, linewidth=0.95,
                                 linestyle=(0, (3, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, fontsize=fs, color=INK2,
                ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def vbox(ax, x, ytop, w, fc, title, lines, fs=6.5, pad=0.012):
    """Box sized to its content: nothing can overflow, because the height is derived from the text."""
    lh = (fs * 1.42) / (ax.figure.get_size_inches()[1] * 72.0)      # one line in axes units
    h = 2 * pad + lh * (1.35 + len(lines))
    y = ytop - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=fc, edgecolor=INK2, linewidth=0.8))
    ax.text(x + w / 2, ytop - pad, title, ha="center", va="top", fontsize=fs + 0.9,
            fontweight="bold", color=INK)
    if lines:
        ax.text(x + w / 2, ytop - pad - lh * 1.35, "\n".join(lines), ha="center", va="top",
                fontsize=fs, color=INK, linespacing=1.42)
    return y


def method():
    fig, ax = canvas(7.4, 6.2)
    y = vbox(ax, 0.03, 0.985, 0.94, DATA, "Public catalogue (real, licence-clean)",
             ["GCAT canonical orbits and design data  \u00b7  CelesTrak operational status, radar cross-section",
              "CelesTrak space weather: observed days strictly before the prediction time only"])
    arrow(ax, (0.5, y), (0.5, y - 0.028))
    y = vbox(ax, 0.03, y - 0.028, 0.94, DATA, "Cohort and administrative censoring, as of origin $O$",
             ["perigee < 2000 km; event = natural decay; controlled endings and objects still in orbit right-censored",
              "learning target  $z = \\log_{10}\\min(T,\\,5\\,\\mathrm{yr})$;  radar cross-section excluded (its availability encodes the outcome)"])
    arrow(ax, (0.5, y), (0.5, y - 0.028))
    ytop = y - 0.028
    cols = [(0.03, 0.225, "train", ["$t_0 < O-7$ yr", "fits the experts"]),
            (0.275, 0.225, "tune-A", ["$O-7 \\leq t_0 < O-5$ yr", "half the launch", "families; search,", "selection, gate check"]),
            (0.52, 0.225, "tune-B", ["the other half", "conformal and", "isotonic calibration"]),
            (0.765, 0.205, "test", ["$O \\leq t_0 <$ period end", "read for scoring only"])]
    ybot = min(vbox(ax, x, ytop, w, GREY, lab, body, fs=6.2) for x, w, lab, body in cols)
    ax.text(0.5, ybot - 0.016, "split by $t_0$; the tuning cohort is divided by a hash of the launch family, so no family spans tune-A and tune-B",
            ha="center", fontsize=6.0, color=INK2)
    ytop = ybot - 0.042
    ya = vbox(ax, 0.03, ytop, 0.295, PHYS, "Physics prior  (pools estimated on train)",
              ["32 Monte-Carlo area-to-mass draws:", "the design value where GCAT has one,",
               "else a class pool estimated on train,", "else the NASA breakup model",
               "King-Hele orbit averages, exact", "quadrature for $e>0.05$", "NRLMSIS 2.1 density at perigee",
               "", "$\\rightarrow$ $q^{\\mathrm{phys}}_\\tau$ and 5 summary features"], fs=6.2)
    yb = vbox(ax, 0.3525, 0.0 + ytop, 0.295, LEARN, "Learned expert  (fitted on train)",
              ["53 features in 7 groups, including", "the 5 physics summaries",
               "candidates: TabM, TabICLv2,", "Kumo-Tabular (2025\u20132026)",
               "12-trial search on tune-A (the two", "in-context models are not tuned),", "5 seeds, averaged",
               "", "$\\rightarrow$ $q^{\\mathrm{ml}}_\\tau$"], fs=6.2)
    yc = vbox(ax, 0.675, ytop, 0.295, FUSE, "Reliability gate  (fitted on train)",
              ["descriptors $r_i$ of the physics prior:", "A/m source, its Monte-Carlo spread,",
               "eccentricity, perigee band, class,", "$P(\\text{at the 5-yr cap})$",
               "fitted on 5-fold out-of-fold", "train predictions (Adam, $L_2$)",
               "", "$\\rightarrow$ $g(r_i)\\in(0,1)$"], fs=6.2)
    arrow(ax, (0.14, ytop + 0.0), (0.14, ytop))
    ybot = min(ya, yb, yc)
    for x in (0.1775, 0.50, 0.8225):
        arrow(ax, (x, ybot), (x, ybot - 0.026))
    ytop = ybot - 0.026
    y = vbox(ax, 0.13, ytop, 0.74, FUSE, "Convex fusion of the two quantile functions",
             ["$q_\\tau(i) = g(r_i)\\, q^{\\mathrm{phys}}_\\tau(i) + (1-g(r_i))\\, q^{\\mathrm{ml}}_\\tau(i)$",
              "a convex combination of monotone quantile functions is monotone, so the output is a valid distribution"], fs=6.3)
    arrow(ax, (0.5, y), (0.5, y - 0.026))
    ytop = y - 0.026
    yl = vbox(ax, 0.03, ytop, 0.45, FUSE, "Calibration on tune-B",
              ["Mondrian conformal intervals, groups =", "object class $\\times$ perigee band",
               "isotonic maps for $P(T\\leq H)$"], fs=6.2)
    yr = vbox(ax, 0.52, ytop, 0.45, EVAL, "Scoring on the untouched test cohort",
              ["pinball loss, MAE, coverage, width, Winkler,", "Harrell's C, AUC-PR, AUC-ROC, Brier, ECE",
               "2,000 cluster bootstraps over launch families"], fs=6.2)
    arrow(ax, (0.48, (ytop + yl) / 2), (0.52, (ytop + yl) / 2))
    ax.text(0.5, min(yl, yr) - 0.022, "Every stage is fitted on data strictly earlier than the cohort it is applied to; the test cohort enters only at the final step.",
            ha="center", fontsize=6.1, color=INK2)
    save(fig, "eaai_method")


def integration():
    fig, ax = canvas(7.4, 4.2)
    box(ax, 0.345, 0.80, 0.31, 0.165, GREY, "Object twin  $s_i(t)$",
        "orbit · design · health evidence\ndebris family · space weather\nconjunction summary")
    box(ax, 0.02, 0.28, 0.28, 0.33, "#dbe8f8", "H  Health",
        "conformal $p$-values per channel\nBenjamini–Hochberg across\nchannels, persistence $k=3$\n\n"
        "out: alarm state,\noperational status")
    box(ax, 0.36, 0.28, 0.28, 0.33, "#fde3d6", "D  Debris",
        "family evidence $\\varphi_F(i,t)$\nconjunction risk: persistence +\nlearned experts, gate, conformal\ncloud survival $E[N(t)]$\n\n"
        "out: $P$(high risk), risk\nquantiles, cloud counts")
    box(ax, 0.70, 0.28, 0.28, 0.33, "#e3f1df", "R  Re-entry",
        "physics prior + learned expert\nreliability gate\nMondrian conformal\nhistory-conditioned A/m\n\n"
        "out: time-to-re-entry\ndistribution, $P(T \\leq H)$")
    arrow(ax, (0.16, 0.61), (0.42, 0.80), "alarms, status", tx=-0.055, ty=0.025, fs=6.0)
    arrow(ax, (0.50, 0.61), (0.50, 0.80))
    arrow(ax, (0.60, 0.80), (0.80, 0.61), "C3:  regime $\\rho_i$", tx=0.075, ty=0.030, fs=6.2)
    arrow(ax, (0.64, 0.50), (0.70, 0.50), "C1:  $\\varphi_F \\rightarrow x_i$", ty=0.042, fs=6.2)
    arrow(ax, (0.70, 0.37), (0.64, 0.37), "C2:  $\\sum_i S_i(t-t_{0,i})$", ty=-0.042, fs=6.2)
    arrow(ax, (0.30, 0.335), (0.36, 0.335), "C4", dashed=True, ty=0.03, fs=6.2)
    rows = [("C1", "debris $\\rightarrow$ re-entry", "breakup- and launch-family statistics known at $t_0$ enter the expert features", "ablation, 2 origins"),
            ("C2", "re-entry $\\rightarrow$ debris", "per-fragment survival summed over a breakup cloud gives its expected population", "61 clouds"),
            ("C3", "health $\\rightarrow$ re-entry", "operational status selects the decay regime; drag forecast withheld when controlled", "13,794 payloads"),
            ("C4", "health $\\times$ debris", "alarm or non-operational status escalates a high-risk conjunction", "no public paired data")]
    y0 = 0.225
    ax.text(0.02, y0, "coupling", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(0.105, y0, "direction", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(0.30, y0, "variable that crosses the boundary", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(0.80, y0, "evaluation", fontsize=6.4, fontweight="bold", color=INK)
    for i, (c, d, v, e) in enumerate(rows):
        y = y0 - 0.045 * (i + 1)
        ax.text(0.02, y, c, fontsize=6.3, color=INK)
        ax.text(0.105, y, d, fontsize=6.3, color=INK2)
        ax.text(0.30, y, v, fontsize=6.3, color=INK2)
        ax.text(0.80, y, e, fontsize=6.3, color=INK2)
    ax.plot([0.02, 0.98], [y0 - 0.018, y0 - 0.018], color=INK2, linewidth=0.6)
    ax.text(0.5, 0.012, "Solid arrows: couplings tested on real data. Dashed: a demonstration, because no public dataset "
            "pairs telemetry with conjunction messages for the same spacecraft.", ha="center", fontsize=6.2, color=INK2)
    save(fig, "eaai_integration")


if __name__ == "__main__":
    method()
    integration()
