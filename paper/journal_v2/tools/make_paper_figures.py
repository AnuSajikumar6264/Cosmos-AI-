"""Artwork for the journal manuscript: one figure per file, vector PDF (fonts embedded) plus a
300 dpi PNG used inside the Word manuscript. Data come only from thesis/evidence/evidence.json.

    python paper/journal_v2/tools/make_paper_figures.py      # -> paper/journal_v2/artwork/Figure_N.{pdf,png}

Figure numbers follow the order of first citation in src/paper.md.
"""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
sys.path.insert(0, os.path.join(ROOT, "thesis", "tools"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType fonts
import matplotlib.pyplot as plt  # noqa: E402
import make_figures as MF  # noqa: E402  (palette, style helpers, evidence access, thesis figures)

ART = os.path.join(PAPER, "artwork")
os.makedirs(ART, exist_ok=True)
v, opt, style = MF.v, MF.opt, MF.style
INK, INK2, MUTED, GRID, SURF, SLOTS = MF.INK, MF.INK2, MF.MUTED, MF.GRID, MF.SURF, MF.SLOTS
FACTORS = [("G", "grid"), ("F", "AR(8) fit data"), ("sigma", "residual scale"), ("P", "calibration pool"),
           ("D", "selection data")]


def save_as(fig, name):
    fig.patch.set_facecolor("white")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(ART, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def fig_decomposition(name):
    """One-factor-at-a-time steps from A0 to A1 (forward) and from A1 to A0 (reverse), four settings."""
    panels = [(1, "target", "conformal_bh", "Mission 1, full channels, conformal-BH (primary)"),
              (1, "target", "threshold", "Mission 1, full channels, threshold"),
              (1, "lightweight", "conformal_bh", "Mission 1, lightweight channels, conformal-BH"),
              (2, "target", "conformal_bh", "Mission 2, full channels, conformal-BH")]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.9), sharex=True, sharey=True)
    for ax, (m, chs, layer, title) in zip(axes.flat, panels):
        pre = f"ph.p1.m{m}.{chs}.{layer}"
        a0, a1 = v(f"{pre}.c0.f05"), v(f"{pre}.c5.f05")
        title = f"{title}\nA0 {a0:.3f}, A1 {a1:.3f}"
        for i, (f, lab) in enumerate(FACTORS):
            for path, off, col, mk in (("forward", -0.14, SLOTS[0], "o"), ("reverse", 0.14, SLOTS[1], "s")):
                d = v(f"{pre}.{path}.{f}.delta_official")
                lo, hi = v(f"{pre}.{path}.{f}.ci95_low"), v(f"{pre}.{path}.{f}.ci95_high")
                ax.plot([lo, hi], [i + off, i + off], color=col, linewidth=1.4)
                ax.scatter([d], [i + off], s=30, color=col, marker=mk, zorder=3, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.axvline(0.05, color=INK2, linewidth=0.6, linestyle=":")
        ax.set_yticks(range(len(FACTORS)), [lab for _, lab in FACTORS], fontsize=8, color=INK)
        style(ax, "change in event-wise F0.5 (95 % CI)" if ax in axes[1] else "")
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=8.2, color=INK, loc="left")
    axes[0, 0].set_ylim(len(FACTORS) - 0.5, -0.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=SLOTS[0], lw=1.4, label="forward path (A0 to A1)"),
                        Line2D([0], [0], marker="s", color=SLOTS[1], lw=1.4, label="reverse path (A1 to A0)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_as(fig, name)


def fig_forward_chain(name):
    """Forward-chaining folds: window protocol (W) against history protocol (H), BH layer."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), gridspec_kw={"width_ratios": [4, 3]})
    for ax, m in zip(axes, (1, 2)):
        pre = f"ph.p2.m{m}.conformal_bh"
        nf = int(v(f"{pre}.n_folds"))
        for i in range(nf):
            w, h = v(f"{pre}.fold{i}.W"), v(f"{pre}.fold{i}.H")
            ax.plot([i - 0.22, i + 0.22], [w, h], color=GRID, linewidth=1.5, zorder=1)
            ax.scatter([i - 0.22], [w], s=34, color=SLOTS[0], zorder=3, edgecolor="white", linewidth=0.8)
            ax.scatter([i + 0.22], [h], s=34, color=SLOTS[1], marker="s", zorder=3, edgecolor="white", linewidth=0.8)
            for x, val, fa in ((i - 0.22, w, v(f"{pre}.fold{i}.W_fa_yr")), (i + 0.22, h, v(f"{pre}.fold{i}.H_fa_yr"))):
                ax.text(x, val + 0.035, f"{val:.2f}\n{fa:.0f} FA/yr", fontsize=6, color=INK2, ha="center", va="bottom")
        labels = [str(v(f"{pre}.fold{i}.start"))[:7] for i in range(nf)]
        ax.set_xticks(range(nf), labels, fontsize=7.5)
        ax.set_xlim(-0.5, nf - 0.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("event-wise F0.5 of the fold", fontsize=8, color=INK2)
        style(ax, "fold start (12-month folds)" if m == 1 else "fold start (3-month folds)", xgrid=False)
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(f"Mission {m}", fontsize=9, color=INK, loc="left")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=SLOTS[0], lw=0, label="W: calibration-window protocol"),
                        Line2D([0], [0], marker="s", color=SLOTS[1], lw=0, label="H: training-history protocol")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_as(fig, name)


def reexport(fn, png_name, name):
    """Re-run a thesis figure function, saving vector PDF + PNG under the artwork name."""
    orig = MF.save

    def save(fig, _n):
        save_as(fig, name)
    MF.save = save
    try:
        fn()
    finally:
        MF.save = orig


def timeline(name):
    src = os.path.join(PAPER, "figures", "protocol_timeline")
    for ext in ("pdf", "png"):
        shutil.copy(f"{src}.{ext}", os.path.join(ART, f"{name}.{ext}"))
    print("wrote", name)


FIGURES = [  # (name, builder)
    ("Figure_1", timeline),
    ("Figure_2", fig_decomposition),
    ("Figure_3", fig_forward_chain),
    ("Figure_S1", lambda n: reexport(MF.esa_selection_effect, "esa_selection_effect.png", n)),   # supplementary
    ("Figure_S2", lambda n: reexport(MF.esa_layers, "esa_layers.png", n)),
    ("Figure_S3", lambda n: reexport(MF.esa_drift_vs_score, "esa_drift_vs_score.png", n)),
]


def main():
    only = set(sys.argv[1:])
    for name, fn in FIGURES:
        if only and name not in only:
            continue
        try:
            fn(name)
        except KeyError as e:
            print(f"skip {name}: evidence key missing ({e})")


if __name__ == "__main__":
    main()
