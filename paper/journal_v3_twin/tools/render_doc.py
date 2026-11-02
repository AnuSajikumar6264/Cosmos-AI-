"""Fill the {{key:format}} placeholders of a Markdown document from build/numbers.json.

    python paper/journal_v3_twin/tools/render_doc.py docs/RESULTS_AND_ANALYSIS.src.md docs/RESULTS_AND_ANALYSIS.md

Same placeholder rules as render.py; a missing value aborts unless --draft is given.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("paper_render", os.path.join(HERE, "render.py"))
_R = importlib.util.module_from_spec(_spec)       # loaded under its own name: render.py itself imports
_spec.loader.exec_module(_R)                       # the thesis's render module as "render"
PAPER, substitute = _R.PAPER, _R.substitute


def main(src, dst, draft=False):
    N = json.load(open(os.path.join(PAPER, "build", "numbers.json")))
    body, missing = substitute(open(src, encoding="utf-8").read(), N)
    if missing and not draft:
        for k in missing:
            print("MISSING:", k)
        raise SystemExit(f"{len(missing)} numbers missing in {src}")
    header = (f"<!-- Generated from {os.path.relpath(src, os.path.dirname(os.path.dirname(PAPER)))} by "
              "paper/journal_v3_twin/tools/render_doc.py; edit the source, not this file. -->\n\n")
    open(dst, "w", encoding="utf-8").write(header + body)
    print(f"wrote {dst}; missing={len(missing)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--draft"]
    main(args[0], args[1], draft="--draft" in sys.argv)
