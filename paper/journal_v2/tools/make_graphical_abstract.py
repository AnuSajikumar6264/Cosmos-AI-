"""Graphical abstract (separate file): drawn at the 13 cm display width, >= 1328 px wide.

    python paper/journal_v2/tools/make_graphical_abstract.py
    -> paper/journal_v2/artwork/Graphical_abstract.{pdf,tiff,png}

Every number is read from thesis/evidence/evidence.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
E = json.load(open(os.path.join(ROOT, "thesis", "evidence", "evidence.json")))
v = lambda k: E[k]["value"]  # noqa: E731
INK, INK2, MUTED, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#eb6834", "#d9d8d2"
CM = 1 / 2.54


def main():
    pre = "ph.p1.m1.target.conformal_bh"
    steps = [("grid", "G"), ("fit\ndata", "F"), ("residual\nscale", "sigma"), ("pool", "P"),
             ("selection\nlabels", "D")]
    a0, a1 = v(f"{pre}.c0.f05"), v(f"{pre}.c5.f05")
    fig = plt.figure(figsize=(13 * CM, 6 * CM))
    fig.text(0.015, 0.95, "Where does an event-level gain come from?", fontsize=9, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.875, "ESA-ADB Mission 1, same AR(8) scorer: one protocol factor switched at a time, A0 to A1 "
             "(frozen plan)", fontsize=6.3, color=INK2, va="top")
    ax = fig.add_axes([0.075, 0.2, 0.55, 0.58])
    labels = ["A0\nwindow"] + [s for s, _ in steps] + ["A1\nhistory"]
    level = a0
    ax.bar(0, a0, color=BLUE, width=0.62)
    ax.text(0, a0 + 0.012, f"{a0:.3f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    for i, (_, f) in enumerate(steps, start=1):
        d = v(f"{pre}.forward.{f}.delta_official")
        if abs(d) < 5e-4:
            ax.plot([i - 0.31, i + 0.31], [level, level], color=MUTED, linewidth=1.2)
            ax.text(i, level + 0.012, "0", ha="center", va="bottom", fontsize=6.3, color=INK2)
        else:
            ax.bar(i, d, bottom=level, color=ORANGE if d > 0 else GREY, width=0.62)
            ax.text(i, max(level, level + d) + 0.012, f"{d:+.3f}", ha="center", va="bottom", fontsize=6.3, color=INK)
        level += d
    ax.bar(len(steps) + 1, a1, color=BLUE, width=0.62)
    ax.text(len(steps) + 1, a1 + 0.012, f"{a1:.3f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=5.6, color=INK)
    ax.set_ylim(0, max(a1, level) * 1.18)
    ax.set_ylabel("event-wise $F_{0.5}$", fontsize=6.3, color=INK2)
    ax.tick_params(axis="y", labelsize=5.6, length=0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    facts = [("Deep scorer", f"TCN vs AR(8): {v('conf.m1.h4.diff_official'):+.3f}\nno gain from capacity"),
             ("Forward in time", f"history protocol wins {v('ph.p2.m1.conformal_bh.n_folds_H_better')} of "
                                 f"{v('ph.p2.m1.conformal_bh.n_folds')} folds:\nnot replicated"),
             ("Other settings", "gain reverses on lightweight\nchannels and on Mission 2 (BH)"),
             ("Adaptive conformal", f"point-rate tuning fails;\nevent tuning {v('ph.p4.m1.ar8.gamma_positive.f05'):.3f}")]
    for i, (h, t) in enumerate(facts):
        y = 0.77 - i * 0.19
        fig.text(0.665, y, h, fontsize=6.4, fontweight="bold", color=INK, va="top")
        fig.text(0.665, y - 0.055, t, fontsize=6.0, color=INK, va="top", linespacing=1.15)
    out = os.path.join(PAPER, "artwork", "Graphical_abstract")
    fig.savefig(out + ".pdf", facecolor="white")
    fig.savefig(out + ".png", dpi=300, facecolor="white")
    fig.savefig(out + ".tiff", dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out + ".tiff").size
    print(f"wrote {out}.pdf/.tiff/.png ({w} x {h} px)")


if __name__ == "__main__":
    main()
