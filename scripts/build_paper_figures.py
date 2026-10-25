"""Figures for paper/manuscript.tex, written to paper/figures/ from the result files.

    python scripts/build_paper_figures.py

selection_landscape.png  training-half CV EW F0.5 over the (q, k) grid of the conformal-BH
                         layer, one panel per mission and score source (training labels only)
test_arms.png            test-half primary metric per arm (seed mean and per-seed points),
                         once the confirmatory test files exist
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cosmos.ai.report import AXIS, GRID, INK, INK2, MUTED, SLOTS, SURFACE  # noqa: E402

PROT = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol")
OUT = os.path.join(ROOT, "paper", "figures")
SEQ = ["#f2f6fc", "#cfe0f5", "#9cc0ec", "#5f97de", "#2a78d6", "#1d5aa6", "#123c70"]


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", p)


def selection_landscape(sources=(("m1_target_ar8", "Mission 1, AR(8) (A1)"),
                                 ("m2_target_ar8", "Mission 2, AR(8) (A1)"))):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    panels = [(t, lbl) for t, lbl in sources
              if os.path.isfile(os.path.join(PROT, "selection", f"{t}.json"))]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.9),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for n, (ax, (t, lbl)) in enumerate(zip(axes, panels)):
        s = json.load(open(os.path.join(PROT, "selection", f"{t}.json")))
        g = pd.DataFrame(s["grid"])
        g = g[g.kind == "conformal_bh"].pivot(index="level", columns="k", values="cv")
        g = g.sort_index(ascending=False)
        im = ax.imshow(g.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        best = np.unravel_index(np.nanargmax(g.values), g.shape)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if v > 0.55 else INK,
                        fontweight="bold" if (i, j) == best else "normal")
        ax.set_xticks(range(g.shape[1]), [str(k) for k in g.columns], fontsize=8, color=INK2)
        ax.set_yticks(range(g.shape[0]), [f"{q:g}" for q in g.index], fontsize=8, color=INK2)
        ax.set_xlabel("persistence k (steps)", fontsize=8, color=INK2)
        if n == 0:
            ax.set_ylabel("BH level q", fontsize=8, color=INK2)
        ax.set_title(f"{lbl}\n{s['n_train_events']} training events", fontsize=9, color=INK,
                     loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cb.set_label("training-half CV event-wise F0.5", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.outline.set_visible(False)
    _save(fig, "selection_landscape.png")


def test_arms(mission: int = 1):
    import matplotlib.pyplot as plt
    rows = []
    for p in glob.glob(os.path.join(PROT, "test", f"m{mission}_target_*.json")):
        for r in json.load(open(p)):
            if r["calibrator"] == "conformal_bh":
                rows.append({"tag": r["tag"], "f": r["anomaly_and_rare/EW_F0.5"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    arm_of = lambda t: ("A1: AR(8), OOF selection" if t.endswith("_ar8") else
                        "A2: XGBoost event scorer" if "_gbt_" in t else
                        "A3: TCN, no residual input" if "no_residual" in t else
                        "A3: TCN, 2 classes" if "_c2_" in t else "A3: TCN event scorer")
    df["arm"] = df.tag.map(arm_of)
    a0 = json.load(open(os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "arms",
                                     f"m{mission}_target_ar8_s0.json")))["rows"]
    a0 = [r for r in a0 if r["calibrator"] == "conformal_bh"][0]["anomaly_and_rare/EW_F0.5"]
    df = pd.concat([pd.DataFrame([{"tag": "a0", "f": a0,
                                   "arm": "A0: AR(8), 9-month selection"}]), df])
    order = ["A0: AR(8), 9-month selection", "A1: AR(8), OOF selection",
             "A2: XGBoost event scorer", "A3: TCN event scorer", "A3: TCN, 2 classes",
             "A3: TCN, no residual input"]
    order = [o for o in order if o in set(df.arm)]
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(order) + 0.9))
    ax.set_facecolor(SURFACE)
    for i, arm in enumerate(order):
        v = df[df.arm == arm].f.to_numpy()
        col = SLOTS[min(i, len(SLOTS) - 1)] if i < len(SLOTS) else MUTED
        ax.barh(i, v.mean(), height=0.55, color=col, edgecolor=SURFACE, linewidth=2)
        if len(v) > 1:
            ax.scatter(v, np.full(len(v), i), s=12, color=INK, zorder=3, linewidths=0)
        ax.text(v.mean() + 0.01, i + 0.3, f"{v.mean():.3f}" + (f" (n={len(v)})" if len(v) > 1
                                                               else ""), fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Mission {mission} test half: corrected event-wise F0.5", fontsize=8,
                  color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    _save(fig, f"test_arms_m{mission}.png")


if __name__ == "__main__":
    selection_landscape()
    for m in (1, 2):
        test_arms(m)
