"""Architecture and workflow diagrams (graphviz) for the thesis -> thesis/figures/*.png

    python thesis/tools/make_diagrams.py

Every box names a module or function that exists in the current code; the labels were checked
against the source (see the thesis chapter text for file references).
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
DOT = "/data/anupa/COSMOS_work/.tools/bin/dot"
os.makedirs(OUT, exist_ok=True)

STYLE = """
  graph [fontname="Helvetica", fontsize=11, bgcolor="#fcfcfb", pad=0.25, nodesep=0.35, ranksep=0.45];
  node  [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled", fillcolor="#ffffff",
         color="#52514e", penwidth=1.0, margin="0.12,0.06"];
  edge  [fontname="Helvetica", fontsize=9, color="#52514e", arrowsize=0.7];
"""
BLUE, ORANGE, GREEN, AMBER, PINK, GREY = "#dbe8f8", "#fbe1d4", "#d2efe3", "#fbecc7", "#f9dfea", "#ecebe6"


def render(name, body):
    src = os.path.join(OUT, f"{name}.dot")
    with open(src, "w") as f:
        f.write("digraph G {\n" + STYLE + body + "\n}\n")
    subprocess.run([DOT, "-Tpng", "-Gdpi=300", src, "-o", os.path.join(OUT, f"{name}.png")], check=True)
    subprocess.run([DOT, "-Tsvg", src, "-o", os.path.join(OUT, f"{name}.svg")], check=True)
    print("wrote", name)


def esa_protocol():
    render("diag_esa_protocol", f"""
  rankdir=TB;
  subgraph cluster_data {{ label="ESA Anomalies Dataset v1.0 (official grids, full target channel sets)"; style="rounded"; color="#c3c2b7";
    split [label="official split\\ntraining half | test half (already used in the exploratory phase)", fillcolor="{GREY}"]; }}
  subgraph cluster_train {{ label="training half only (labels allowed)"; style="rounded"; color="#c3c2b7";
    folds [label="K = 5 contiguous blocks, 1-day purge\\n(oof.fold_bounds)", fillcolor="{BLUE}"];
    ar8 [label="ridge AR(8) forecaster per channel\\nfitted on the other blocks (oof.ar8_oof)\\n→ OOF non-conformity s(t,c)", fillcolor="{BLUE}"];
    feats [label="7 causal features per (t, c)\\n(event_model.features)", fillcolor="{BLUE}"];
    scorers [label="event scorers trained out of fold\\nA2: XGBoost (event_gbt)   A3: causal TCN (event_model)", fillcolor="{GREEN}"];
    select [label="alarm-layer selection by cross-fitting\\nover the two halves of the training half\\n(oof_benchmark.select: q, k on a fixed grid)", fillcolor="{AMBER}"]; }}
  freeze [label="pre-registration + SHA-256 freeze of 28 files\\n(PREREG_TELEMETRY_V1.md, freeze.json)", shape=note, fillcolor="{PINK}"];
  subgraph cluster_test {{ label="test half (scored once per arm)"; style="rounded"; color="#c3c2b7";
    test [label="gated single test pass\\n(oof_benchmark.test: refuses if any frozen file changed)", fillcolor="{ORANGE}"];
    stats [label="paired hierarchical block bootstrap, Holm\\n(prereg_analysis.py stats → CONFIRMATORY.json)", fillcolor="{ORANGE}"]; }}
  split -> folds -> ar8 -> feats -> scorers;
  ar8 -> select [label=" A1 scores"];
  scorers -> select [label=" A2, A3 scores"];
  select -> freeze [style=dashed, arrowhead=none];
  freeze -> test [label=" hash check"];
  select -> test [label=" chosen (q, k)"];
  split -> test [label=" test scores from models\\n fitted on the whole training half", style=dotted];
  test -> stats;
""")


def alarm_layer():
    render("diag_alarm_layer", f"""
  rankdir=LR;
  s [label="scores s(t,c)\\n(any source)", fillcolor="{GREY}"];
  pool [label="nominal OOF training\\nscores per channel\\n= calibration pool", fillcolor="{BLUE}"];
  p [label="conformal p-value\\np(t,c) = (1 + #pool ≥ s) / (n_c + 1)", fillcolor="{BLUE}"];
  bh [label="Benjamini–Hochberg\\nacross channels at each t\\n(level q)", fillcolor="{BLUE}"];
  hot [label="hot cells H(t,c)", fillcolor="{GREY}"];
  per [label="persistence: ≥ k consecutive\\nsteps with any hot channel", fillcolor="{AMBER}"];
  merge [label="merge runs closer\\nthan 1 h", fillcolor="{AMBER}"];
  attr [label="global alarm + attributed\\nchannels (hot inside the run)", fillcolor="{ORANGE}"];
  s -> p; pool -> p; p -> bh -> hot -> per -> merge -> attr;
""")


def event_tcn():
    render("diag_event_tcn", f"""
  rankdir=TB;
  x [label="input window (B, C, 7, T)\\nT = 2048 steps, all C target channels", fillcolor="{GREY}"];
  inp [label="1×1 conv 7 → 64  +  learned channel embedding", fillcolor="{BLUE}"];
  blk [label="8 residual blocks, dilation 2^i (i = 0…7)\\n[causal conv k3 → ChannelNorm → SiLU → dropout] × 2", fillcolor="{BLUE}"];
  mix [label="every 2 blocks: cross-channel mixing\\nh ← h + W · mean_c(h)   (permutation-equivariant)", fillcolor="{GREEN}"];
  head [label="1×1 conv 64 → 3 logits per (t, c)\\nnominal | anomaly | rare nominal event", fillcolor="{BLUE}"];
  loss [label="class-weighted cross-entropy (1, 5, 5)\\non the last T − 1021 steps; gaps ignored", fillcolor="{AMBER}"];
  out [label="event score = log P(anomaly ∪ rare) − log P(nominal)", fillcolor="{ORANGE}"];
  x -> inp -> blk -> head -> out; blk -> mix [dir=both, style=dashed]; head -> loss [style=dashed];
""")


def ai_layer():
    render("diag_ai_layer", f"""
  rankdir=LR;
  subgraph cluster_d {{ label="cosmos.ai.data"; style="rounded"; color="#c3c2b7";
    dl [label="download.py\\nchecksummed, resumable,\\nmanifests", fillcolor="{GREY}"];
    pp [label="preprocess_telemetry.py\\npreprocess_catalogue.py", fillcolor="{GREY}"];
    va [label="validate.py\\nDATASET_REPORT", fillcolor="{GREY}"]; }}
  subgraph cluster_t {{ label="cosmos.ai.telemetry"; style="rounded"; color="#c3c2b7";
    esa [label="ESA-ADB: forecasting, alarms,\\noof, event_model, event_gbt,\\noof_benchmark, esa_metrics (official)", fillcolor="{BLUE}"];
    ops [label="OPSSAT-AD: opssat, robustness", fillcolor="{BLUE}"];
    lmon [label="learned_monitor (synthetic)", fillcolor="{BLUE}"]; }}
  subgraph cluster_v {{ label="cosmos.ai.vision"; style="rounded"; color="#c3c2b7";
    sg [label="spotgeo, heatmap,\\ntrain_spotgeo (official scorer)", fillcolor="{GREEN}"];
    syn [label="cosmos_synthetic, metrics", fillcolor="{GREEN}"]; }}
  subgraph cluster_r {{ label="re-entry"; style="rounded"; color="#c3c2b7";
    lmk [label="survival/landmark", fillcolor="{AMBER}"];
    sur [label="surrogate", fillcolor="{AMBER}"]; }}
  subgraph cluster_x {{ label="cross-cutting"; style="rounded"; color="#c3c2b7";
    unc [label="uncertainty, evaluation,\\nexplain", fillcolor="{PINK}"];
    exp [label="experiment (Run records)", fillcolor="{PINK}"];
    fus [label="fusion, registry, report", fillcolor="{PINK}"]; }}
  dl -> pp -> va;
  pp -> esa; pp -> ops; pp -> lmk; pp -> sg;
  esa -> unc [style=dashed]; lmk -> unc [style=dashed]; sur -> fus [style=dashed]; lmon -> fus [style=dashed];
""")


def overall():
    render("diag_overall", f"""
  rankdir=LR; compound=true;
  subgraph cluster_h {{ label="cosmos.health  (Feature 1)"; style="rounded"; color="#c3c2b7";
    tel [label="telemetry.py\\n16-channel simulator\\n+ fault injection", fillcolor="{BLUE}"];
    mon [label="monitor.py\\nHealthMonitor (robust z, EWMA)", fillcolor="{BLUE}"];
    haz [label="hazard.py\\nassess_hazard: subsystem →\\nbreak-up type, altitude, σ log B", fillcolor="{BLUE}"];
    real [label="realdata.py\\nSMAP/MSL, ESA-AD scoring\\n(stand-alone)", fillcolor="{GREY}"]; }}
  subgraph cluster_r {{ label="cosmos.reentry  (Feature 3)"; style="rounded"; color="#c3c2b7";
    bu [label="breakup.py\\nsimplified SBM cloud", fillcolor="{AMBER}"];
    mc [label="montecarlo.py\\nfootprint, Gaussian vs\\nconformal zones", fillcolor="{AMBER}"];
    rk [label="risk.py + population.py\\nE_c, land probability", fillcolor="{AMBER}"];
    phys [label="atmosphere.py (US76)\\ndrag.py, propagator.py (3-DOF)", fillcolor="{GREY}"]; }}
  subgraph cluster_d {{ label="cosmos.debris  (Feature 2; not in run_pipeline)"; style="rounded"; color="#c3c2b7";
    cat [label="catalogue.py\\nSATCAT, break-up clouds,\\nKaplan–Meier", fillcolor="{GREEN}"];
    life [label="lifetime.py\\ndrag-decay lifetime,\\ncensored A/M fit", fillcolor="{GREEN}"];
    det [label="scene.py, detect.py, yolo.py\\nsynthetic frame pairs,\\nregistration + DoG, TinyYOLO", fillcolor="{GREEN}"]; }}
  subgraph cluster_ai {{ label="cosmos.ai  (COSMOS-AI)"; style="rounded"; color="#c3c2b7";
    lm [label="telemetry.learned_monitor\\n(backend='learned')", fillcolor="{PINK}"];
    sur [label="fusion.surrogate_footprint\\n(footprint_backend='surrogate')", fillcolor="{PINK}"];
    studies [label="real-data studies:\\nESA-ADB, OPSSAT-AD, SpotGEO,\\nre-entry landmark", fillcolor="{PINK}"]; }}
  tel -> mon -> haz -> bu -> mc -> rk;
  phys -> mc [style=dashed];
  lm -> mon [style=dashed, label=" replaces (optional)"];
  sur -> mc [style=dashed, label=" replaces (optional)"];
  cat -> life [style=dashed];
  life -> phys [style=dotted, arrowhead=none, label=" shares US76"];
""")


def pipeline_flow():
    render("diag_pipeline_flow", f"""
  rankdir=TB;
  a [label="simulate_telemetry(hours=24, dt=30 s, faults, seed)\\n→ 2880 × 16 samples", fillcolor="{BLUE}"];
  b [label="monitor: HealthMonitor().fit().score()   or   LearnedHealthMonitor\\n→ anomaly score A(t), subsystem degradation D_s(t), first alarm", fillcolor="{BLUE}"];
  c [label="health = D_s at the last sample;  score = A at the last sample", fillcolor="{GREY}"];
  d [label="assess_hazard(score, health)\\ndriver = argmax_s D_s → break-up type, altitude h_bu, σ log10 B\\n(also P_frag and σ_state: displayed only, not used downstream)", fillcolor="{AMBER}"];
  e [label="sample_cloud(parent_mass = 1800 kg, kind)\\n→ fragments (L_c, A/M, m, B, Δv), mass closure", fillcolor="{AMBER}"];
  f [label="footprint: monte_carlo_footprint (RK4 ensemble, US76, tumbling C_D)\\nor surrogate_footprint (5-member MLP ensemble, range fallback)", fillcolor="{AMBER}"];
  g [label="assess_footprint(fp, population)\\n→ per-fragment E_c, risk band, land probability, nearest city", fillcolor="{ORANGE}"];
  h [label="zone_comparison (optional, with_zones=True)\\nGaussian ellipse vs split-conformal zones", fillcolor="{GREY}"];
  r [label="PipelineResult.summary()", shape=note, fillcolor="{PINK}"];
  a -> b -> c -> d -> e -> f -> g -> r; f -> h [style=dashed]; h -> r [style=dashed];
""")


if __name__ == "__main__":
    overall()
    pipeline_flow()
    esa_protocol()
    alarm_layer()
    event_tcn()
    ai_layer()
