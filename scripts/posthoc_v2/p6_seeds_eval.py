"""P6: selection and test scoring of the seed-extension runs (A2, A3 seeds 5-9, Mission 1), then
the H4/H5 comparisons with all available seeds, using the exact bootstrap (P3).

    python scripts/posthoc_v2/p6_seeds_eval.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ROOT, TMP, provenance, save_json, out_path  # noqa: E402
import exact_metric as XM  # noqa: E402

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.alarms import Calibrator  # noqa: E402
from cosmos.ai.telemetry.esa_eval import false_alarm_stats, headline, score_alarms  # noqa: E402
from cosmos.ai.telemetry.forecasting import MissionData  # noqa: E402
from cosmos.ai.data.preprocess_telemetry import official_test_labels  # noqa: E402

PROT = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol")
M = 1


def main():
    prov = provenance(__file__)
    base = os.path.join(OUT, "p6_seeds")

    def rd(*parts):
        d = os.path.join(base, *parts)
        os.makedirs(d, exist_ok=True)
        return d
    OB.result_dir = rd
    D = MissionData(M, "target")
    ts = D.test.timestamps()
    lab = official_test_labels(M, D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    ev_dir = os.path.join(TMP, "benchmarks", "telemetry_esa_adb", "event_model")
    names = {"A2": "m1_target_gbt_ar8_s{}", "A3": "m1_target_event_ar8_all_c3_s{}"}
    runs_out = {}
    for arm, pat in names.items():
        runs_out[arm] = []
        for seed in range(5, 10):
            d = os.path.join(ev_dir, pat.format(seed))
            tag = f"{pat.format(seed)}_event"
            saved_sel = os.path.join(base, "selection", f"{tag}.json")
            saved_al = {k: os.path.join(base, "alarms", f"{tag}_{k}_global.npy") for k in ("conformal_bh", "threshold")}
            if os.path.isfile(saved_sel) and all(os.path.isfile(p) for p in saved_al.values()):
                # scored in an earlier pass whose score arrays were on the cleared /tmp: rebuild the
                # row from the saved selection and alarms (exact metric, P3)
                sel = json.load(open(saved_sel))
                for kind in ("conformal_bh", "threshold"):
                    c = sel["chosen"][kind]
                    g = np.unpackbits(np.load(saved_al[kind]))[:len(ts)].astype(bool)
                    row = {"tag": tag, "calibrator": kind, "level": c["level"], "k": c["k"], "cv": c["cv_EW_F0.5"],
                           "anomaly_and_rare/EW_F0.5": XM.f_beta(XM.event_table(g, ts, lab)),
                           **false_alarm_stats(ts, g, lab), "from_saved_alarms": True}
                    runs_out[arm].append(row)
                    print(f"{arm} {tag} {kind}: F0.5 {row['anomaly_and_rare/EW_F0.5']:.4f} (saved alarms)", flush=True)
                continue
            if not os.path.isfile(os.path.join(d, "event_test.npy")):
                runs_out[arm].append({"tag": tag, "dir": d, "status": "missing"})
                print(f"{arm} {tag}: missing", flush=True)
                continue
            src = f"event:{d}"
            # the frozen selection is deterministic (P8); reuse a saved result for the same source
            saved = os.path.join(base, "selection", f"{os.path.basename(d.rstrip('/'))}_event.json")
            sel = json.load(open(saved)) if os.path.isfile(saved) else None
            if sel is None or sel.get("source") != src:
                sel = OB.select(M, "target", src)
            S_tr, S_te, tag = OB.load_source(M, "target", src)
            for kind in ("conformal_bh", "threshold"):
                c = sel["chosen"][kind]
                cal = Calibrator(kind, c["level"], c["k"], c["merge"], rule_s=D.train.rule_s).fit(
                    np.asarray(S_tr, np.float32), D.Ltr)
                g, ch = cal.alarms(np.asarray(S_te, np.float32))
                np.save(out_path("p6_seeds", "alarms", f"{tag}_{kind}_global.npy"), np.packbits(g))
                sc = score_alarms(M, ts, ch, D.channels, lab, per_channel=False)
                row = {"tag": tag, "calibrator": kind, "level": c["level"], "k": c["k"], "cv": c["cv_EW_F0.5"],
                       **{f"anomaly_and_rare/{k}": v for k, v in headline(sc, "anomaly_and_rare").items()},
                       **{f"anomaly/{k}": v for k, v in headline(sc, "anomaly").items()},
                       **false_alarm_stats(ts, g, lab)}
                runs_out[arm].append(row)
                print(f"{arm} {tag} {kind}: F0.5 {row['anomaly_and_rare/EW_F0.5']:.4f}", flush=True)
    # comparisons with all seeds (original 0-4 + new), conformal-BH, exact bootstrap
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prereg_analysis as PA
    orig = {k: [PA.tag_of(M, "target", s) for s in v] for k, v in PA.arms(M, "target").items()}

    def tab(tag, frozen=True):
        p = os.path.join(PROT if frozen else os.path.join(OUT, "p6_seeds"), "alarms", f"{tag}_conformal_bh_global.npy")
        return XM.event_table(np.unpackbits(np.load(p))[:len(ts)].astype(bool), ts, lab)
    tabs = {"A1": [tab(t) for t in orig["A1"]],
            "A2": [tab(t) for t in orig["A2"]] + [tab(r["tag"], False) for r in runs_out["A2"]
                                                   if r.get("calibrator") == "conformal_bh"],
            "A3": [tab(t) for t in orig["A3"]] + [tab(r["tag"], False) for r in runs_out["A3"]
                                                   if r.get("calibrator") == "conformal_bh"]}
    rng = np.random.default_rng(20261024)
    comps = []
    for h, x, y in (("H4: A3 - A1", "A3", "A1"), ("H5: A3 - A2", "A3", "A2"), ("X1: A2 - A1", "A2", "A1")):
        r = XM.compare(tabs[x], tabs[y], rng)
        r.pop("replicates")
        comps.append({"comparison": h, **r, "n_seeds_A": len(tabs[x]), "n_seeds_B": len(tabs[y])})
        print(comps[-1], flush=True)
    vals = {k: [XM.f_beta(t) for t in v] for k, v in tabs.items()}
    save_json({**prov, "new_runs": runs_out, "comparisons": comps,
               "seed_values": vals, "n_seeds": {k: len(v) for k, v in vals.items()}}, "p6_seeds", "summary.json")


if __name__ == "__main__":
    main()
