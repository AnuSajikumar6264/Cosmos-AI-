"""Write paper/tables/*.tex from the result files (no number is typed by hand).

    python scripts/build_paper_tables.py

baselines.tex     published DMLR values, our reproduction of GlobalSTD, label-free baselines
arms.tex          the pre-registered arms
confirmatory.tex  H4-H6 (Mission 1) and the Mission 2 replication, if CONFIRMATORY.json exists
ablations.tex     secondary calibrators, ablations, ACI comparator, if the test files exist
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
OUT = os.path.join(ROOT, "paper", "tables")
ESA = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb")
PROT = os.path.join(ESA, "oof_protocol")

#: DMLR 2026, Table 4: corrected event-wise F0.5, anomalies + rare events
DMLR = {(1, "lightweight"): {"global_std3": "0.001", "global_std5": "0.253",
                             "best": ("Telemanom-ESA-Pruned", "0.786")},
        (1, "target"): {"global_std3": "<0.001", "global_std5": "0.003",
                        "best": ("Telemanom-ESA-Pruned", "0.061")},
        (2, "lightweight"): {"global_std3": "0.007", "global_std5": "0.075",
                             "best": ("Windowed iForest", "0.949")},
        (2, "target"): {"global_std3": "0.008", "global_std5": "0.064", "best": ("PCC", "0.079")}}
#: published values are kept as printed (3 significant digits), never re-rounded


def f(x, d=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.4f}" if 0 < abs(x) < 0.01 and d == 3 else f"{x:.{d}f}"


def baselines():
    from cosmos.ai.telemetry.esa_benchmark import collect
    df = collect()
    key = "anomaly_and_rare/EW_F0.5"
    get = lambda m, ch, enc, cal="native": df[(df.mission == m) & (df.channels == ch)
                                              & (df.encoder == enc) & (df.calibrator == cal)
                                              & ~df.telecommands.fillna(False).astype(bool)]
    rows = []
    for name, enc, cal, pub in (("GlobalSTD3 (official)", "global_std3", "native", "global_std3"),
                                ("GlobalSTD5 (official)", "global_std5", "native", "global_std5"),
                                ("Out-of-limit, nominal training range", "out_of_limits",
                                 "native", None),
                                ("Out-of-limit, full training range", "out_of_limits_all",
                                 "native", None),
                                ("COSMOS monitor (AR(8), EWMA, fixed rule)", "cosmos_monitor",
                                 "native", None),
                                ("A0: AR(8) + conformal-BH, 9-month selection$^\\dagger$", "ar8",
                                 "conformal_bh", None)):
        cells = []
        for m, ch in ((1, "target"), (1, "lightweight"), (2, "target"), (2, "lightweight")):
            r = get(m, ch, enc, cal)
            ours = f(float(r[key].iloc[0])) if len(r) else "--"
            if pub:
                ours = f"{ours} ({f(DMLR[(m, ch)][pub])})"
            cells.append(ours)
        rows.append((name, cells))
    best = [f"{f(DMLR[k]['best'][1])}" for k in ((1, "target"), (1, "lightweight"),
                                                 (2, "target"), (2, "lightweight"))]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Corrected event-wise $F_{0.5}$ (anomalies and rare events) on the "
             r"official test halves. GlobalSTD rows show our run and, in parentheses, the "
             r"value published in DMLR Table~4. $^\dagger$Exploratory: selected on the last "
             r"9 months of the training half.}\label{tab:baselines}",
             r"\begin{tabular}{lcccc}\toprule",
             r" & \multicolumn{2}{c}{Mission 1} & \multicolumn{2}{c}{Mission 2}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Detector & full (58) & lightweight & full (47) & lightweight\\\midrule",
             r"Best published (DMLR) & " + " & ".join(best) + r"\\"]
    lines += [f"{n} & " + " & ".join(c) + r"\\" for n, c in rows]
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def arms():
    return r"""\begin{table}[t]\centering\small
\caption{Pre-registered arms (freeze \texttt{09a98f5e160e}). All share the alarm layer of
Section~\ref{sec:alarm}.}\label{tab:arms}
\begin{tabular}{llll}\toprule
Arm & Scores & Selection of $(q,k)$ & Seeds\\\midrule
A0 & AR(8), fitted on 2000--2005 (M1) & cross-fit on 9-month calibration window & --\\
A1 & AR(8), out of fold ($K=5$) & cross-fit on whole training half & --\\
A2 & XGBoost event scorer, out of fold & cross-fit on whole training half & 5\\
A3 & TCN event scorer, out of fold & cross-fit on whole training half & 5\\
\bottomrule\end{tabular}\end{table}"""


def confirmatory():
    p = os.path.join(PROT, "CONFIRMATORY.json")
    if not os.path.isfile(p):
        return None
    res = json.load(open(p))
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Confirmatory comparisons, primary metric, conformal-BH layer. Values "
             r"are official seed means; the interval and $p$ come from the paired hierarchical "
             r"block bootstrap. Mission~1 is the pre-registered family (Holm); Mission~2 is a "
             r"replication.}\label{tab:confirmatory}",
             r"\begin{tabular}{llcccccl}\toprule",
             r"Mission & Test & A & B & $\Delta$ & 95\,\% CI & $p_{\mathrm{Holm}}$ & Verdict\\"
             r"\midrule"]
    for m, r in res["missions"].items():
        for t in r["tests"]:
            lines.append(f"{m} & {t['comparison'].split(':')[0]} & {f(t['A_official'])} & "
                         f"{f(t['B_official'])} & {t['diff_official']:+.3f} & "
                         f"[{t['ci95_low']:+.3f}, {t['ci95_high']:+.3f}] & "
                         f"{f(t.get('p_holm'), 4) if 'p_holm' in t else '--'} & "
                         f"{t['verdict']}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


ARM_NAMES = [("_ar8$", "A1: AR(8), OOF"), ("_gbt_ar8_s", "A2: XGBoost"),
             ("_event_ar8_all_c3_s", "A3: TCN"),
             ("_event_ar8_no_residual_c3_s", "A3 without residual inputs"),
             ("_event_ar8_all_c2_s", "A3 with 2 classes")]


def ablations():
    import re
    rows = []
    for p in sorted(glob.glob(os.path.join(PROT, "test", "*.json"))):
        for r in json.load(open(p)):
            rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["mission"] = df.tag.str[1].astype(int)
    df["channels"] = np.where(df.tag.str.contains("lightweight"), "lightweight", "full")
    df["arm"] = [next((n for pat, n in ARM_NAMES if re.search(pat, t)), t) for t in df.tag]
    cals = [("threshold", "threshold"), ("conformal_bh", "conformal-BH"),
            ("conformal_bh_rolling", "rolling BH"), ("aci", "ACI"),
            ("aci_event", "ACI + events")]
    key = "anomaly_and_rare/EW_F0.5"
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Test-half corrected event-wise $F_{0.5}$ for every arm and alarm "
             r"layer (mean $\pm$ sd over seeds; $n$ seeds). The conformal-BH column is the "
             r"pre-registered primary layer, and the other columns are secondary. ACI is the "
             r"point-level adaptive conformal comparator. FA/yr: false alarms per year under "
             r"conformal-BH.}\label{tab:ablations}",
             r"\begin{tabular}{lll" + "c" * (len(cals) + 2) + r"}\toprule",
             "M & set & Arm & " + " & ".join(c[1] for c in cals) + r" & FA/yr & $n$\\\midrule"]
    order = {n: i for i, (_, n) in enumerate(ARM_NAMES)}
    for (m, ch, arm), g in sorted(df.groupby(["mission", "channels", "arm"]),
                                  key=lambda x: (x[0][0], x[0][1], order.get(x[0][2], 9))):
        cells = []
        for c, _ in cals:
            v = g[g.calibrator == c][key]
            cells.append("--" if v.empty else (f"{v.mean():.3f}" + (f" $\\pm$ {v.std():.3f}"
                                                                      if len(v) > 1 else "")))
        fa = g[g.calibrator == "conformal_bh"].false_alarms_per_year
        n = int((g.calibrator == "conformal_bh").sum())
        lines.append(f"{m} & {ch} & {arm} & " + " & ".join(cells)
                     + f" & {fa.mean():.1f} & {n}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("baselines", baselines), ("arms", arms), ("confirmatory", confirmatory),
                     ("ablations", ablations)):
        s = fn()
        if s is None:
            print(f"{name}: inputs not there yet")
            continue
        with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
            fh.write(s + "\n")
        print(f"{name}: written")


if __name__ == "__main__":
    main()
