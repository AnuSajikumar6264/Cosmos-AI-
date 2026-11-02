"""Render the digital-twin manuscript: numbers -> Markdown -> Word (and LaTeX).

    python paper/journal_v3_twin/tools/build_numbers.py
    python paper/journal_v3_twin/tools/render.py

The sources in src/ use {{key:format}} placeholders filled from build/numbers.json (written by
build_numbers.py from the result files).  A placeholder without a value aborts the build.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
sys.path.insert(0, os.path.join(ROOT, "thesis", "tools"))
sys.path.insert(0, os.path.join(ROOT, "paper", "journal_v2", "tools"))
import render as thesis_render  # noqa: E402  (PANDOC, XREF paths, fmt)
import render_paper as v2  # noqa: E402  (restyle: table widths, equation tables)

PARTS = ["front.md", "part1_intro_methods.md", "part2_results.md", "part3_discussion.md", "back.md"]
PH = re.compile(r"\{\{([A-Za-z0-9_.\-=]+)(?::([^}]+))?\}\}")


def substitute(text, N):
    missing = []

    def rep(m):
        k, spec = m.group(1), m.group(2)
        if k not in N or N[k]["value"] is None:
            missing.append(k)
            return "[MISSING]"
        v = N[k]["value"]
        if spec == "pct":
            return f"{100 * float(v):.1f} %"
        if spec and spec.endswith("%") and spec != "%":
            return format(float(v) * 100, spec[:-1]) + " %"
        if spec:
            return format(v, spec) if not isinstance(v, str) else v
        return str(v)
    out = PH.sub(rep, text)
    return out, sorted(set(missing))


def main(draft=False):
    N = json.load(open(os.path.join(PAPER, "build", "numbers.json")))
    src = "\n\n".join(open(os.path.join(PAPER, "src", p), encoding="utf-8").read() for p in PARTS
                      if os.path.isfile(os.path.join(PAPER, "src", p)))
    body, missing = substitute(src, N)
    if missing and not draft:
        for k in missing:
            print("MISSING:", k)
        raise SystemExit(f"{len(missing)} numbers missing; build aborted (use --draft for a layout check)")
    os.makedirs(os.path.join(PAPER, "build"), exist_ok=True)
    md = os.path.join(PAPER, "build", "manuscript.md")
    open(md, "w", encoding="utf-8").write(body)
    common = ["--from", "markdown+tex_math_dollars+pipe_tables+lists_without_preceding_blankline", "--filter", thesis_render.XREF, "--citeproc",
              "--bibliography", os.path.join(PAPER, "refs.bib"), "--csl", os.path.join(HERE, "elsevier-harvard.csl"),
              "--resource-path", PAPER, "--metadata-file", os.path.join(HERE, "crossref.yaml")]
    out = os.path.join(PAPER, "build", "manuscript.docx")
    r = subprocess.run([thesis_render.PANDOC, md, *common, "--reference-doc", os.path.join(HERE, "reference.docx"),
                        "--number-sections", "--columns=10000", "-o", out], capture_output=True, text=True, cwd=PAPER)
    if r.returncode:
        print(r.stderr)
        raise SystemExit("pandoc failed (docx)")
    for line in [x for x in r.stderr.splitlines() if x.strip()][:20]:
        print("pandoc:", line)
    eq_ids = re.findall(r"\{#eq:([A-Za-z0-9_\-]+)\}", body)
    appendix = {str(i + 1): f"{e[0]}.{e[1:]}" for i, e in enumerate(eq_ids) if re.fullmatch(r"[A-Z]\d+", e)}
    v2.restyle(out, appendix)
    tex = os.path.join(PAPER, "build", "manuscript.tex")
    r = subprocess.run([thesis_render.PANDOC, md, *common, "--standalone", "--number-sections",
                        "--include-in-header", os.path.join(HERE, "latex_header.tex"), "-o", tex],
                       capture_output=True, text=True, cwd=PAPER)
    if r.returncode:
        print(r.stderr[:2000])
        print("LaTeX export failed")
    words = len(re.sub(r"(?s)^---.*?---", "", body.split("# References")[0]).split())
    print(f"wrote {out} and {tex}; about {words} words before the references; missing={len(missing)}")


if __name__ == "__main__":
    main(draft="--draft" in sys.argv)
