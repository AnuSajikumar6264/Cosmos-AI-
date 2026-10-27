"""Write paper/journal_v2/src/supplementary.md (placeholders only; rendered by render_paper.py).

Tables and figures are numbered S1, S2, ... in order of appearance, and the numbers are written
to src/supplementary_labels.json so that the manuscript can cite them as [S:label]. Sections whose
evidence does not exist yet are left out, so the file can be regenerated while post-hoc analyses
are still running.

    python paper/journal_v2/tools/make_supplementary.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
E = json.load(open(os.path.join(ROOT, "thesis", "evidence", "evidence.json")))


def has(*keys):
    return all(k in E and E[k]["value"] is not None for k in keys)


L = []
w = L.append
LABELS = {}
COUNT = {"Table": 0, "Fig.": 0}


def caption(kind, label, title, text):
    """Numbered caption paragraph: 'Table S3. Title. Text' (tables: above; figures: below)."""
    COUNT[kind] += 1
    num = f"S{COUNT[kind]}"
    LABELS[label] = f"{kind} {num}"
    return f"**{kind} {num}.** **{title}** {text}"


def table(label, title, text, header, rows, note=None):
    w(f": {caption('Table', label, title, text)}\n")
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(r) + " |")
    w("")
    if note:
        w(f"::: {{custom-style=\"Table Note\"}}\nNote: {note}\n:::\n")


def figure(label, png, title, text, width="85%"):
    w(f"![{caption('Fig.', label, title, text)}](artwork/{png}){{width={width}}}\n")


w("---\ntitle: \"Supplementary Material\"\nsubtitle: \"Beyond model complexity: a pre-registered evaluation of event-level alarm protocols for spacecraft telemetry anomaly detection\"\n---\n")
w("This file accompanies the manuscript. Every number is inserted from the study's evidence file (`thesis/evidence/evidence.json`), which records the result file each value comes from. Section S4 reports the validation analyses of the second frozen plan (`POSTHOC_PLAN_V2.md`, SHA-256 prefix `{{ph.plan_sha256}}`, frozen {{ph.plan_written}}). Abbreviations: A0–A3, study arms; BH, Benjamini–Hochberg; ACI, adaptive conformal inference; M1, M2, Missions 1 and 2; FA, false alarms.\n")

# ---------------------------------------------------------------- S1
w("# S1. Pre-registration, deviation log and validation plan\n")
w("The following documents are provided as separate files in the repository and in the archived release:\n")
w("- `research_design/preregistration/PREREG_TELEMETRY_V1.md`: the pre-registration, frozen as `{{prereg.freeze_id}}`, with {{prereg.frozen_files}} files hashed;")
w("- `research_design/preregistration/DEVIATIONS.md`: every departure from the pre-registration, with its timing;")
w("- `research_design/preregistration/freeze_09a98f5e160e.json` and `freeze.json`: the original freeze and the re-issued one (`{{prereg.refreeze_id}}`);")
w("- `research_design/preregistration/POSTHOC_PLAN_V2.md` and `posthoc_v2_freeze.json`: the validation plan (P1–P8), frozen before any of its analyses was run.\n")

# ---------------------------------------------------------------- S2
w("# S2. Results reported on ESA-ADB\n")
table("s-lit", "Results reported on ESA-ADB.", "Primary scores as published; the protocols differ, so the numbers are not comparable.",
      ["Study", "Protocol", "Method", "Reported score"],
      [["@kotowski2026esaadb", "official split; full and lightweight channels", "TimeEval baselines, detector-specific thresholds",
        "M1 full {{lit.dmlr.m1.full.best:.3f}}; M2 full {{lit.dmlr.m2.full.best:.3f}}; M1 lightweight {{lit.dmlr.m1.lw.best:.3f}}"],
       ["@sahin2026tccad", "official code; {{lit.tccad.m1_channels}} / {{lit.tccad.m2_channels}} channels; {{lit.tccad.seeds}} seeds; pre-registered",
        "telecommand conditioning; split, sliding and adaptive conformal thresholds", "conditioning negative; point-level ACI collapses the event score"],
       ["@nguyen2026", "own preprocessing; uncorrected event-wise metrics", "supervised CNN and graph networks against unsupervised detectors", "not comparable"],
       ["@brancato2027ast", "Mission 1, hierarchical protocol (as stated in the abstract)", "similarity search (SMED); quantile GP regression with retraining",
        "M1 full {{lit.brancato.smed_full:.3f}} (SMED)"],
       ["@alcarria2026", "monthly incremental retraining", "LSTM autoencoder; PCA, k-means and LOF; adaptive thresholds",
        "M1 {{lit.alcarria.m1_f05:.3f}}; M2 {{lit.alcarria.m2_f05:.3f}}"],
       ["@allegrini2026", "Kaggle split of Mission 1, hidden test", "supervised hierarchical ensemble", "{{lit.allegrini.private:.3f}} (private leaderboard)"],
       ["@heydari2026", "two missions, {{lit.heydari.shared_channels}} shared channels", "threshold transfer under alert budgets", "policy diagnostics"],
       ["This study", "official split; full target channels; pre-registered", "AR(8), XGBoost and TCN under one event-level layer",
        "M1 {{pre.m1.A1.conformal_bh.f05.mean:.3f}} (AR(8)), {{pre.m1.A2.conformal_bh.f05.mean:.3f}} (trees)"]],
      "GP, Gaussian process; LOF, local outlier factor; SMED, the similarity-search detector of Brancato et al.")

w("# S3. Models and pre-registered arms\n")
table("s-models", "Model settings.", "No setting was tuned on test data; the supervised scorers received no hyperparameter search.",
      ["Model", "Setting", "Value"],
      [["AR(8)", "order; forecast horizon; maximum fitted rows", "{{cfg.oof.order}}; {{cfg.oof.horizon}} steps; {{cfg.oof.max_fit:int}}"],
       ["AR(8)", "out-of-fold blocks; purge", "5 contiguous blocks; {{cfg.evm.purge_days:g}} d on either side"],
       ["AR(8)", "score", "absolute forecast error / max(residual scale, 0.05)"],
       ["XGBoost (A2)", "trees; depth; learning rate", "{{cfg.gbt.rounds}}; {{cfg.gbt.max_depth}}; {{cfg.gbt.eta}}"],
       ["XGBoost (A2)", "row and column subsampling; minimum child weight", "{{cfg.gbt.subsample}}; {{cfg.gbt.colsample_bytree}}; {{cfg.gbt.min_child_weight}}"],
       ["XGBoost (A2)", "training rows; classes; class weights", "{{cfg.gbt.n_rows:int}} (half labelled); {{cfg.gbt.num_class}}; {{cfg.evm.class_weights}}"],
       ["TCN (A3)", "levels; kernel; width; receptive field", "{{cfg.evm.levels}}; {{cfg.evm.kernel}}; {{cfg.evm.d_model}}; {{evm.receptive_field}} steps"],
       ["TCN (A3)", "cross-channel mixing; parameters (M1)", "every {{cfg.evm.mix_every}} levels; {{evm.m1.params:int}}"],
       ["TCN (A3)", "optimiser; steps; learning rate; weight decay; dropout", "AdamW, one-cycle; {{cfg.evm.steps:int}}; {{cfg.evm.lr}}; {{cfg.evm.weight_decay}}; {{cfg.evm.dropout}}"],
       ["TCN (A3)", "batch; window; share of windows with a labelled sample", "{{cfg.evm.batch}}; {{cfg.evm.window:int}} steps; {{cfg.evm.event_fraction}}"],
       ["A2, A3", "per-step features", "asinh z; log(1 + s); its EWMA over 1 h, 12 h, 3 d; deviation from trailing 1 d mean; forecast indicator"]])
for m in (1, 2):
    chan = m == 1
    hdr = ["Arm", "Precision", "Recall", "Alarming precision", "ADTQC", "Affiliation", "Rare-only runs"] + (["Channel", "Subsystem"] if chan else [])
    rows = []
    for arm in ("A0", "A1", "A2", "A3"):
        p = f"pre.m{m}.{arm}.conformal_bh"
        cells = [f"{{{{{p}.p.mean:.3f}}}}", f"{{{{{p}.r.mean:.3f}}}}", f"{{{{{p}.alarm_prec.mean:.3f}}}}",
                 f"{{{{{p}.adtqc.mean:.3f}}}}", f"{{{{{p}.aff.mean:.3f}}}}", f"{{{{{p}.rare_only.mean:.0f}}}}"]
        if chan:
            cells += [f"{{{{{p}.chan.mean:.3f}}}}", f"{{{{{p}.subsys.mean:.3f}}}}"]
        rows.append([arm] + cells)
    table(f"s-secondary-m{m}", f"Secondary metrics, Mission {m}.", "Conformal-BH layer, test half, seed means.", hdr, rows,
          "ADTQC, alarm detection timing quality; affiliation, affiliation-based $F_{0.5}$; rare-only runs, alarm runs touching only rare events"
          + ("; channel and subsystem, channel- and subsystem-aware $F_{0.5}$." if chan else "."))
rows = []
names = {"A3nr": "A3, no residual inputs", "A3c2": "A3, two classes", "A1lw": "A1, lightweight"}
for m, arms in ((1, ("A1", "A2", "A3", "A3nr", "A3c2", "A1lw")), (2, ("A1", "A2", "A3", "A1lw"))):
    for a in arms:
        p = f"pre.m{m}.{a}"
        rows.append([str(m), names.get(a, a), f"{{{{{p}.threshold.f05.mean:.3f}}}}", f"{{{{{p}.conformal_bh.f05.mean:.3f}}}}",
                     f"{{{{{p}.conformal_bh_rolling.f05.mean:.3f}}}}", f"{{{{{p}.aci.f05.mean:.4f}}}}", f"{{{{{p}.aci_event.f05.mean:.3f}}}}"])
table("s-layers", "Alarm layers on identical scores.", "Test-half event-wise $F_{0.5}$, seed means.",
      ["Mission", "Arm", "Threshold", "Conformal-BH", "Rolling BH", "ACI", "ACI + events"], rows,
      "ACI and ACI + events are the point-level comparator as pre-registered ($\\gamma$ tuned to a point-wise rate).")
figure("s-seleff", "Figure_S1.png", "Generality of the training-half protocol's effect.",
       "Training-half protocol minus calibration-window protocol for AR(8), test-half event-wise $F_{0.5}$, by mission, channel set and alarm layer. The ringed point is the pre-registered H6 comparison; all other points are exploratory single runs.", "70%")
figure("s-layersfig", "Figure_S2.png", "Alarm layers on identical scores.",
       "Test-half event-wise $F_{0.5}$ of the same scores under the threshold, conformal-BH and rolling layers (seed means). Coinciding markers mean equal values; ACI is omitted because all its values are below 0.002.", "60%")
rows = []
for s in range(5):
    a2 = f"pre.seed.m1_target_gbt_ar8_s{s}_event.conformal_bh"
    a3 = f"pre.seed.m1_target_event_ar8_all_c3_s{s}_event.conformal_bh"
    rows.append([str(s), f"{{{{{a2}.f05:.3f}}}}", f"{{{{{a2}.fa_yr:.1f}}}}", f"{{{{{a3}.f05:.3f}}}}", f"{{{{{a3}.fa_yr:.1f}}}}"])
table("s-seeds", "Per-seed results of the supervised arms.", "Mission 1, conformal-BH layer, test half (pre-registered seeds 0–4).",
      ["Seed", "A2 $F_{0.5}$", "A2 FA / yr", "A3 $F_{0.5}$", "A3 FA / yr"], rows)
rows = []
for name, enc, cal in (("AR(8), conformal-BH", "ar8", "conformal_bh"), ("AR(8), threshold", "ar8", "threshold"),
                       ("out-of-limit rule", "ool", "native"), ("GlobalSTD5", "gstd5", "native")):
    rows.append([name, f"{{{{explo.m1.full.{enc}.{cal}.f05.mean:.4f}}}}", f"{{{{explo.m1.lw.{enc}.{cal}.f05.mean:.3f}}}}",
                 f"{{{{explo.m2.full.{enc}.{cal}.f05.mean:.3f}}}}", f"{{{{explo.m2.lw.{enc}.{cal}.f05.mean:.3f}}}}"])
table("s-explo", "Exploratory phase.", "AR(8) with the layer selected on the calibration window, and the label-free baselines; test-half $F_{0.5}$ (before the pre-registration).",
      ["Detector", "M1 full", "M1 lightweight", "M2 full", "M2 lightweight"], rows)
figure("s-drift", "Figure_S3.png", "Calibration shift against test score (exploratory).",
       "For each supervised run on Mission 1 (BH layer), the ratio of the nominal test exceedance rate to the cross-validation rate at the $1-10^{-4}$ quantile, against test-half event-wise $F_{0.5}$.", "60%")

# ---------------------------------------------------------------- S3 post hoc
w("# S4. Validation analyses (P1–P8)\n")
w("All analyses in this section were specified and frozen before they were run, after the confirmatory analysis. They use test halves that had been used before.\n")
if has("ph.p3.n_checks"):
    w("**P3, exact bootstrap.** The block-decomposable metric of Appendix A was compared with the official value on {{ph.p3.n_checks}} stored arm–metric pairs; the largest absolute difference was {{ph.p3.max_abs_diff:.2g}} (criterion {{ph.p3.passed}}).\n")
CELLS = (("c0", "S0", "cal", "cal", "G0"), ("c1", "S0", "cal", "cal", "G1"), ("c2", "S½", "cal", "cal", "G1"),
         ("c3", "S1", "cal", "cal", "G1"), ("c4", "S1", "half", "cal", "G1"), ("c5", "S1", "half", "half", "G1"),
         ("r1", "S1", "half", "half", "G0"), ("r2", "S1", "half", "cal", "G0"), ("r3", "S1", "cal", "cal", "G0"),
         ("r4", "S½", "cal", "cal", "G0"))
STEPS = (("G", "grid"), ("F", "AR(8) fit data"), ("sigma", "residual scale"), ("P", "calibration pool"), ("D", "selection data"))
for m, chs, layer in ((1, "target", "conformal_bh"), (1, "target", "threshold"), (1, "lightweight", "conformal_bh"),
                      (2, "target", "conformal_bh"), (2, "target", "threshold"), (2, "lightweight", "conformal_bh")):
    pre = f"ph.p1.m{m}.{chs}.{layer}"
    if not has(f"{pre}.c0.f05"):
        continue
    lay = "conformal-BH" if layer == "conformal_bh" else "threshold"
    rows = [[c, S, P, D, G, f"{{{{{pre}.{c}.f05:.3f}}}}", f"{{{{{pre}.{c}.fa_yr:.1f}}}}", f"{{{{{pre}.{c}.level:g}}}}", f"{{{{{pre}.{c}.k}}}}"]
            for c, S, P, D, G in CELLS]
    table(f"p1-m{m}-{chs}-{layer}", f"P1 decomposition cells, Mission {m}, {chs} channels, {lay} layer.",
          f"Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{{{{pre}.reproduction}}}}.",
          ["Cell", "Scores", "Pool", "Selection data", "Grid", "$F_{0.5}$", "FA / yr", "Level", "$k$"], rows,
          "Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.")
    rows = []
    for f, name in STEPS:
        rows.append([name] + [f"{{{{{pre}.{p}.{f}.delta_official:+.3f}}}} [{{{{{pre}.{p}.{f}.ci95_low:+.3f}}}}, {{{{{pre}.{p}.{f}.ci95_high:+.3f}}}}]"
                              for p in ("forward", "reverse")] + [f"{{{{{pre}.verdict.{f}}}}}"])
    table(f"p1-steps-m{m}-{chs}-{layer}", f"P1 steps, Mission {m}, {chs} channels, {lay} layer.",
          "Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.",
          ["Factor", "Forward path", "Reverse path", "Verdict"], rows,
          "The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.")
for m in (1, 2):
    pre = f"ph.p2.m{m}.conformal_bh"
    if not has(f"{pre}.n_folds"):
        continue
    rows = []
    for layer in ("conformal_bh", "threshold"):
        pl = f"ph.p2.m{m}.{layer}"
        for i in range(int(E[f"{pl}.n_folds"]["value"])):
            f = f"{pl}.fold{i}"
            rows.append(["conformal-BH" if layer == "conformal_bh" else "threshold", f"{{{{{f}.start}}}}", f"{{{{{f}.n_events}}}}",
                         f"{{{{{f}.W:.3f}}}}", f"{{{{{f}.H:.3f}}}}", f"{{{{{f}.diff:+.3f}}}}", f"{{{{{f}.W_fa_yr:.1f}}}}", f"{{{{{f}.H_fa_yr:.1f}}}}"])
    table(f"p2-m{m}", f"P2 forward chaining, Mission {m}.",
          "Per fold: events of the fold, $F_{0.5}$ and false alarms per year for the calibration-window protocol W and the training-history protocol H.",
          ["Layer", "Fold start", "Events", "W $F_{0.5}$", "H $F_{0.5}$", "H − W", "W FA / yr", "H FA / yr"], rows,
          f"Pooled over folds (BH): H − W = {{{{{pre}.pooled.diff:+.3f}}}} [{{{{{pre}.pooled.ci95_low:+.3f}}}}, {{{{{pre}.pooled.ci95_high:+.3f}}}}]. "
          f"Replication criterion met: {{{{{pre}.replicates}}}}.")
p4 = sorted({k.split(".")[2] + "." + k.split(".")[3] for k in E if k.startswith("ph.p4.m") and k.endswith(".any_gamma.f05")})
if p4:
    rows = []
    for t in p4:
        p = f"ph.p4.{t}"
        rows.append([t.replace(".ar8", " A1").replace(".gbt_ar8_s", " A2 seed ").replace("_event", "").replace("m1", "M1").replace("m2", "M2"),
                     f"{{{{{p}.any_gamma.alpha:g}}}}, {{{{{p}.any_gamma.gamma:g}}}}, {{{{{p}.any_gamma.k}}}}", f"{{{{{p}.any_gamma.f05:.3f}}}}",
                     f"{{{{{p}.gamma_positive.alpha:g}}}}, {{{{{p}.gamma_positive.gamma:g}}}}, {{{{{p}.gamma_positive.k}}}}", f"{{{{{p}.gamma_positive.f05:.3f}}}}",
                     f"{{{{{p}.gamma_zero.f05:.3f}}}}"])
    table("s-p4", "P4, ACI tuned on the event objective.",
          "Chosen $(\\alpha, \\gamma, k)$ and test-half $F_{0.5}$ for the best setting overall, the best with $\\gamma > 0$, and $\\gamma = 0$ (a fixed per-channel conformal threshold).",
          ["Arm", "Best: $\\alpha$, $\\gamma$, $k$", "$F_{0.5}$", "Best with $\\gamma>0$", "$F_{0.5}$", "$\\gamma=0$ $F_{0.5}$"], rows)
for m in (1, 2):
    if has(f"ph.p5.m{m}.A1.mean"):
        rows = []
        for arm in ("A1", "A2", "A3"):
            p = f"ph.p5.m{m}.{arm}"
            if has(f"{p}.mean"):
                rows.append([arm, f"{{{{{p}.n}}}}", f"{{{{{p}.mean:.3f}}}}", f"{{{{{p}.std:.3f}}}}", f"{{{{{p}.min:.3f}}}}", f"{{{{{p}.max:.3f}}}}", f"{{{{{p}.fa_yr:.1f}}}}"])
        table(f"s-p5-m{m}", f"P5, registered anomaly-score arm, Mission {m}.",
              "Anomaly-only test-half $F_{0.5}$ with the layer selected on the anomaly-only objective.",
              ["Arm", "Runs", "Mean", "sd", "Min", "Max", "FA / yr"], rows, "A1 is deterministic; A2 and A3 use their anomaly log-odds.")
if has("ph.p6.m1.A2.f05.n"):
    rows = [[arm, f"{{{{ph.p6.m1.{arm}.f05.n}}}}", f"{{{{ph.p6.m1.{arm}.f05.mean:.3f}}}}", f"{{{{ph.p6.m1.{arm}.f05.std:.3f}}}}",
             f"{{{{ph.p6.m1.{arm}.f05.min:.3f}}}}", f"{{{{ph.p6.m1.{arm}.f05.max:.3f}}}}"] for arm in ("A2", "A3")]
    rows += [[f"{h.upper()}: {lab}", "", f"{{{{ph.p6.m1.{h}.diff:+.3f}}}}", "", f"[{{{{ph.p6.m1.{h}.ci95_low:+.3f}}}}, {{{{ph.p6.m1.{h}.ci95_high:+.3f}}}}]", f"p = {{{{ph.p6.m1.{h}.p_boot:.3f}}}}"]
             for h, lab in (("h4", "A3 − A1"), ("h5", "A3 − A2"), ("x1", "A2 − A1"))]
    table("s-p6", "P6, seed extension.", "Mission 1, conformal-BH layer: seed statistics with all available seeds, and the comparisons with the exact bootstrap (unadjusted, exploratory).",
          ["Arm or comparison", "Seeds", "Mean $F_{0.5}$ / difference", "sd", "Min / 95 % CI", "Max / $p$"], rows)
if has("ph.p7.m1.A1.delay_h_median"):
    rows = []
    for m in (1, 2):
        for arm in ("A0", "A1", "A2", "A3"):
            p = f"ph.p7.m{m}.{arm}"
            rows.append([str(m), arm, f"{{{{{p}.detected:.1f}}}} / {{{{{p}.n_events:.0f}}}}", f"{{{{{p}.detected_anomalies:.1f}}}}",
                         f"{{{{{p}.detected_rare:.1f}}}}", f"{{{{{p}.delay_h_median:.2f}}}} [{{{{{p}.delay_h_q25:.2f}}}}, {{{{{p}.delay_h_q75:.2f}}}}]",
                         f"{{{{{p}.detected_within_1h:.2f}}}}", f"{{{{{p}.detected_within_24h:.2f}}}}",
                         f"{{{{{p}.runs_per_detected_event_median:.1f}}}}", f"{{{{{p}.false_alarms_per_1000_nominal_h:.3f}}}}"])
    table("s-p7", "P7, operational metrics.", "Test halves, conformal-BH layer; seed means for A2 and A3.",
          ["Mission", "Arm", "Detected / events", "Anomalies", "Rare events", "Delay, median [IQR] (h)", "Within 1 h", "Within 24 h",
           "Runs per detected event", "FA per 1000 nominal h"], rows,
          "Delay is first-alarm time minus event start for detected events; IQR, interquartile range.")
if has("ph.p8.m1.passed"):
    w("**P8, provenance.** Re-running the frozen selection for the A1 arms reproduced the saved choice and every grid value: Mission 1 {{ph.p8.m1.passed}} (largest difference {{ph.p8.m1.max_abs_cv_diff:.2g}}), Mission 2 {{ph.p8.m2.passed}} (largest difference {{ph.p8.m2.max_abs_cv_diff:.2g}}).\n")

# ---------------------------------------------------------------- S4
w("# S5. Code, data and reproduction\n")
w("The repository contains the frozen code of the pre-registered study, the validation scripts (`scripts/posthoc_v2/`), the evidence builder (`thesis/tools/build_evidence.py`) and the manuscript build (`paper/journal_v2/tools/`). The ESA Anomaly Dataset is obtained from Zenodo (DOI 10.5281/zenodo.12528696). Derived score and alarm arrays will be deposited with the archived release.\n")
w("# References\n\n::: {#refs}\n:::\n")
open(os.path.join(PAPER, "src", "supplementary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(LABELS, open(os.path.join(PAPER, "src", "supplementary_labels.json"), "w"), indent=1)
print("wrote src/supplementary.md;", ", ".join(f"{k}={v}" for k, v in LABELS.items()))
