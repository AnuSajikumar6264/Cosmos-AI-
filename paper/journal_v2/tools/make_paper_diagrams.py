"""Protocol-and-timeline diagram for the journal manuscript.

    python paper/journal_v2/tools/make_paper_diagrams.py   # -> journal_v2/figures/protocol_timeline.png

The order of the boxes follows the reconstructed timeline (thesis/audit/STAGE1_AUDIT.md, section 3).
The boxes describe what happened; they contain no results.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#c3c2b7"
FILL = {"explo": "#ecebe6", "pre": "#f9dfea", "train": "#dbe8f8", "model": "#d2efe3",
        "sel": "#fbecc7", "test": "#fbe1d4"}

# (x, y, w, h, text, fill); the figure is 10 x 3.6 units
BOXES = {
    "ex":  (0.2, 2.1, 2.6, 1.1, "Exploratory phase\n128 test-half result rows\n(mission, channels, encoder, layer, seed);\nlayer selected on a short window", "explo"),
    "pr":  (3.5, 2.1, 2.6, 1.1, "Pre-registration\nH4–H6, arms, grids, seeds,\nminimum effect, Holm correction", "pre"),
    "fz":  (6.8, 2.1, 2.9, 1.1, "SHA-256 freeze\ndocument + 27 code files;\ntest function refuses to run\nif a frozen file changed", "pre"),
    "oof": (0.2, 0.35, 2.2, 1.1, "Training half only\nout-of-fold AR(8) scores\n(5 blocks, 1-day purge)", "train"),
    "sc":  (2.75, 0.35, 2.2, 1.1, "Event scorers\ntrained out of fold:\nXGBoost (A2), TCN (A3)", "model"),
    "sel": (5.3, 0.35, 2.1, 1.1, "Alarm-layer selection\ncross-fit on the two halves\nof the training half", "sel"),
    "te":  (7.75, 0.35, 1.95, 1.1, "Confirmatory pass\neach new arm scored once\non the previously used test\nhalf; block bootstrap, Holm", "test"),
}
EDGES = [("ex", "pr", "Mission 1 failed"), ("pr", "fz", ""), ("oof", "sc", ""), ("sc", "sel", ""),
         ("sel", "te", "")]


def centre(b, side):
    x, y, w, h = BOXES[b][:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]


fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis("off")
fig.patch.set_facecolor(SURF)
for key, (x, y, w, h, text, kind) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=FILL[kind], edgecolor=INK2, linewidth=0.9))
    title, rest = text.split("\n", 1)
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.47, rest, ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.3)
for a, b, lab in EDGES:
    p = FancyArrowPatch(centre(a, "r"), centre(b, "l"), arrowstyle="-|>", mutation_scale=10, color=INK2, linewidth=0.9)
    ax.add_patch(p)
    if lab:
        (x0, y0), (x1, y1) = centre(a, "r"), centre(b, "l")
        ax.text((x0 + x1) / 2, y0 + 0.08, lab, ha="center", va="bottom", fontsize=7, color=INK2)
# freeze -> training-half work: an elbow from the bottom of the freeze box to the left of the lower row
x_f, y_f = centre("fz", "b")
x_o, y_o = centre("oof", "t")
y_mid = 1.78
ax.plot([x_f, x_f, x_o], [y_f, y_mid, y_mid], color=INK2, linewidth=0.9)
ax.add_patch(FancyArrowPatch((x_o, y_mid), (x_o, y_o), arrowstyle="-|>", mutation_scale=10, color=INK2,
                             linewidth=0.9))
ax.text(5.3, y_mid + 0.06, "code frozen before any new arm was scored on test", ha="center", va="bottom",
        fontsize=7, color=INK2)
# A1 bypasses the event scorers
ax.annotate("", xy=(5.6, 0.35), xytext=(1.6, 0.35), arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.8,
            mutation_scale=9, connectionstyle="arc3,rad=0.25"))
ax.text(3.85, 0.0, "A1 uses the AR(8) scores directly", ha="center", fontsize=7, color=INK2)
import matplotlib as _mpl
_mpl.rcParams.update({"pdf.fonttype": 42})
for _ext, _kw in (("png", {"dpi": 300}), ("pdf", {})):
    fig.savefig(os.path.join(OUT, f"protocol_timeline.{_ext}"), bbox_inches="tight", facecolor="white", **_kw)
print("wrote protocol_timeline.png")
