"""P6 seed extension: train A2 (XGBoost) or A3 (TCN) seeds with the frozen code.

    python scripts/posthoc_v2/seeds_train.py gbt 5
    python scripts/posthoc_v2/seeds_train.py tcn 5 --device xpu:1

The frozen training functions are called unchanged; only their output directory is redirected
from outputs/benchmarks/.../event_model/ (on the nearly full /data disk) to /tmp/posthoc_v2/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TMP, provenance, save_json  # noqa: E402

from cosmos.ai.telemetry import event_gbt, event_model  # noqa: E402


def tmp_output_dir(*parts):
    d = os.path.join(TMP, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("gbt", "tcn"))
    ap.add_argument("seed", type=int)
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--device", default="xpu:0")
    a = ap.parse_args()
    prov = provenance(__file__)
    event_gbt.output_dir = tmp_output_dir
    event_model.output_dir = tmp_output_dir
    if a.kind == "gbt":
        d = event_gbt.run(a.mission, "target", seed=a.seed)
    else:
        d = event_model.run(event_model.EventConfig(mission=a.mission, seed=a.seed, device=a.device))
    save_json({**prov, "kind": a.kind, "seed": a.seed, "mission": a.mission, "dir": d,
               "meta": json.load(open(os.path.join(d, "meta.json")))},
              "seeds", f"train_m{a.mission}_{a.kind}_s{a.seed}.json")
    print("done", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
