"""Parallel driver for the gated test pass (procedural only; see preregistration/DEVIATIONS.md).

    python scripts/prereg_test_parallel.py          # run in several processes at once

It runs the same frozen ``oof_benchmark.test`` as ``prereg_analysis.py test``. Each worker
claims an arm atomically (an O_EXCL lock file next to the result), and an arm that already
has a result is never scored again. Each arm is therefore still scored exactly once, and only
the wall time changes.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402
from scripts.prereg_analysis import arms, tag_of  # noqa: E402


def main():
    OB.check_freeze()
    todo = [(m, "target", s) for m in (1, 2) for v in arms(m).values() for s in v]
    todo += [(m, "lightweight", "ar8") for m in (1, 2)]
    for m, ch, src in todo:
        t = tag_of(m, ch, src)
        out = os.path.join(OB.result_dir("test"), f"{t}.json")
        if os.path.isfile(out):
            continue
        try:
            os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        print(f"TEST M{m} {t}", flush=True)
        OB.test(m, ch, src)


if __name__ == "__main__":
    main()
