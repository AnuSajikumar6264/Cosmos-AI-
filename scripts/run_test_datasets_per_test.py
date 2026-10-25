#!/usr/bin/env python
"""Run each check of tests/test_datasets.py separately, so that the one pre-existing pandas-3
error (test_safe_unpickler) does not hide the other results.  Usage: python scripts/run_test_datasets_per_test.py ."""
import sys, os, traceback
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.join(sys.argv[1], "tests"))
import test_datasets as T
for name in [n for n in dir(T) if n.startswith("test_")]:
    print(f"\n== {name}")
    try:
        getattr(T, name)()
    except Exception as e:
        T.RESULTS.append({"check": name, "status": "ERROR", "note": repr(e)[:200]})
        print("  [ERROR]", repr(e)[:300])
import pandas as pd
df = pd.DataFrame(T.RESULTS)
print("\nSUMMARY", df.status.value_counts().to_dict())
