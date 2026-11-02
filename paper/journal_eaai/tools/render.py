"""Render the Engineering Applications of Artificial Intelligence manuscript.

    python paper/journal_eaai/tools/render.py [--draft]

Builds, from the same ``numbers.json`` as the long-form manuscript so that no value can diverge:

* ``build/manuscript.docx``        single column, the submission artefact (the journal desk-rejects
                                   two-column submissions);
* ``build/manuscript.tex``/``.pdf``  single column, Elsevier ``elsarticle``;
* ``build/manuscript_twocol.pdf``  two column, same source, for preprints and thesis chapters.

Numbers come from ``paper/journal_v3_twin/build/numbers.json``; a missing key aborts the build.
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
V3 = os.path.join(ROOT, "paper", "journal_v3_twin")
sys.path.insert(0, os.path.join(ROOT, "thesis", "tools"))
import render as thesis_render  # noqa: E402  (PANDOC, XREF)

# the long-form manuscript's renderer has the same module name, so load it by path
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("v3_render", os.path.join(V3, "tools", "render.py"))
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
substitute = _v3.substitute

SECTIONS = ["00_front.md", "01_introduction.md", "02_related_work.md", "03_objectives.md", "04_architecture.md",
            "05_debris.md", "06_reentry.md", "07_health.md", "08_couplings.md", "09_datasets.md", "10_setup.md",
            "11_results.md", "12_ablations.md", "13_discussion.md", "14_conclusion.md", "15_back.md"]


def pandoc(md, out, *extra, cwd=None, natbib=False):
    """``natbib=True`` for the LaTeX targets: Elsevier's class loads natbib, which conflicts with
    citeproc's LaTeX bibliography, so the TeX builds use natbib + BibTeX (the journal's own
    workflow) while the Word build uses citeproc."""
    cite = ["--natbib"] if natbib else ["--citeproc", "--csl", os.path.join(HERE, "elsevier-harvard.csl")]
    common = ["--from", "markdown+tex_math_dollars+pipe_tables+lists_without_preceding_blankline",
              "--filter", thesis_render.XREF, *cite, "--bibliography", os.path.join(PAPER, "refs.bib"),
              "--resource-path", PAPER,
              "--metadata-file", os.path.join(HERE, "crossref.yaml"), "--number-sections"]
    r = subprocess.run([thesis_render.PANDOC, md, *common, *extra, "-o", out],
                       capture_output=True, text=True, cwd=cwd or PAPER)
    if r.returncode:
        print(r.stderr[:3000])
        raise SystemExit(f"pandoc failed for {os.path.basename(out)}")
    return r


def longtable_to_fullwidth(tex):
    """Convert pandoc's longtable environments into full-width table* floats.

    ``longtable`` refuses to run in two-column mode, and every table here is wider than one
    column, so the two-column variant needs ``table*``.  Mechanical and verified by compiling.
    """
    src = open(tex, encoding="utf-8").read()
    pat = re.compile(r"\\begin\{longtable\}(\[[^\]]*\])?\{([^}]*)\}(.*?)\\end\{longtable\}", re.S)

    def one(m):
        spec, body = m.group(2), m.group(3)
        cap = ""
        mc = re.search(r"\\caption\{(.*?)\}\\tabularnewline\s*", body, re.S)
        if mc:
            cap = mc.group(1)
            body = body[:mc.start()] + body[mc.end():]
        for drop in (r"\\endfirsthead", r"\\endhead", r"\\endlastfoot", r"\\endfoot"):
            body = re.sub(drop, "", body)
        # pandoc repeats the header for the running head; keep only the first occurrence
        body = re.sub(r"(\\toprule\\noalign\{\}\s*)+", r"\\toprule\\noalign{}\n", body, count=0)
        parts = body.split(r"\midrule\noalign{}")
        if len(parts) > 2:
            body = parts[0] + r"\midrule\noalign{}" + parts[-1]
        capline = ("\\caption{" + cap + "}\n") if cap else ""
        return ("\\begin{table*}[t]\n\\centering\n\\footnotesize\n" + capline +
                "\\begin{tabular}{" + spec + "}\n" + body.strip() + "\n\\end{tabular}\n\\end{table*}")
    out, n = pat.subn(one, src)
    # the multi-panel figures are unreadable at column width, so make every float full width
    out = out.replace("\\begin{figure}", "\\begin{figure*}").replace("\\end{figure}", "\\end{figure*}")
    out = out.replace("\\begin{figure*}[", "\\begin{figure*}[t]%")
    open(tex, "w", encoding="utf-8").write(out)
    return n


def tectonic(tex):
    exe = shutil.which("tectonic") or "/data/anupa/COSMOS_work/.tools/bin/tectonic"
    d = os.path.dirname(tex)
    for sub in ("artwork", "artwork_eaai"):
        link = os.path.join(d, sub)
        if os.path.islink(link):
            os.remove(link)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(PAPER, sub)), link)
    r = subprocess.run([exe, "-X", "compile", "--keep-logs", os.path.basename(tex)],
                       capture_output=True, text=True, cwd=d)
    log = os.path.join(d, os.path.basename(tex).replace(".tex", ".log"))
    miss = sum(1 for ln in open(log, errors="ignore") if "Missing character" in ln) if os.path.isfile(log) else -1
    print(f"  {os.path.basename(tex)}: rc={r.returncode} missing_glyphs={miss}")
    if r.returncode:
        print(r.stdout[-2500:])
    return r.returncode == 0


def main(draft=False):
    N = json.load(open(os.path.join(V3, "build", "numbers.json")))
    parts = []
    for p in SECTIONS:
        f = os.path.join(PAPER, "src", p)
        if os.path.isfile(f):
            parts.append(open(f, encoding="utf-8").read())
    raw = "\n\n".join(parts)
    # splice generated tables: a marker with no table aborts, so no section can lose its evidence
    tdir = os.path.join(PAPER, "build", "tables")
    used, absent = set(), set()

    def table(m):
        k = m.group(1)
        f = os.path.join(tdir, f"{k}.md")
        if not os.path.isfile(f):
            absent.add(k)
            return f"[MISSING TABLE {k}]"
        used.add(k)
        return open(f, encoding="utf-8").read().rstrip()
    raw = re.sub(r"\{\{TABLE:([a-z0-9_]+)\}\}", table, raw)
    if absent:
        raise SystemExit("tables not generated: " + ", ".join(sorted(absent)) +
                         " — run paper/journal_eaai/tools/make_tables.py")
    print(f"tables spliced: {len(used)} ({', '.join(sorted(used))})")
    body, missing = substitute(raw, N)
    if missing and not draft:
        for k in missing:
            print("MISSING:", k)
        raise SystemExit(f"{len(missing)} numbers missing; build aborted (use --draft to check layout)")
    b = os.path.join(PAPER, "build")
    os.makedirs(b, exist_ok=True)
    md = os.path.join(b, "manuscript.md")
    open(md, "w", encoding="utf-8").write(body)
    words = len(body.split())
    print(f"source: {len(parts)} sections, about {words} words before substitution losses")

    pandoc(md, os.path.join(b, "manuscript.docx"), "--reference-doc", os.path.join(HERE, "reference.docx"),
           "--columns=10000")
    hdr = os.path.join(HERE, "latex_header.tex")
    if not os.path.lexists(os.path.join(b, "refs.bib")):
        os.symlink(os.path.join(PAPER, "refs.bib"), os.path.join(b, "refs.bib"))
    for name, opts in (("manuscript", ["-V", "classoption=3p,times"]),
                       ("manuscript_twocol", ["-V", "classoption=3p,times,twocolumn"])):
        tex = os.path.join(b, f"{name}.tex")
        pandoc(md, tex, "--standalone", "--template", os.path.join(HERE, "elsarticle.latex"), *opts,
               "--include-in-header", hdr, natbib=True)
        if "twocol" in name:
            print(f"  converted {longtable_to_fullwidth(tex)} longtables to full-width floats")
        tectonic(tex)
    print("wrote", b)


if __name__ == "__main__":
    main(draft="--draft" in sys.argv)
