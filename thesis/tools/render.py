"""Render the thesis: substitute evidence placeholders, then build Word (and Markdown) output.

    python thesis/tools/render.py            # -> thesis/build/COSMOS_Thesis.docx (+ .md)

Placeholders in thesis/src/*.md:

``{{key}}``          the evidence value as stored (strings verbatim, numbers with 3 decimals)
``{{key:.2f}}``      any Python format spec
``{{key:int}}``      integer with thousands separator
``{{key:pct1}}``     fraction shown as a percentage with 1 decimal (0.0123 -> 1.2 %)
``{{key:sci1}}``     scientific notation with 1 decimal (2.9e-07)

A missing key aborts the build: the thesis cannot report a number that has no evidence entry.
The keys actually used are listed, with value and source, in an appendix written to
thesis/src/99_evidence_appendix.md (generated; do not edit) and in build/used_keys.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.dirname(HERE)
ROOT = os.path.dirname(TH)
PANDOC = "/data/anupa/COSMOS_work/.tools/pandoc311/bin/pandoc"
XREF = "/data/anupa/COSMOS_work/.tools/bin/pandoc-crossref"
PH = re.compile(r"\{\{([A-Za-z0-9_.\-]+)(?::([^}]+))?\}\}")
TOCFIELD = ('```{=openxml}\n<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Right-click and choose '
            '"Update Field" (or press F9) to build the table of contents.</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>\n```')
PAGEBREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'


def numbering(md):
    """Map crossref ids to the numbers they get in the thesis (sections, tables, figures)."""
    ids, cnt = {}, [0, 0, 0, 0]
    in_code = False
    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        m = re.match(r"^(#{1,4}) (.*?)\s*(\{[^}]*\})?\s*$", line)
        if in_code or not m:
            continue
        attrs = m.group(3) or ""
        if ".unnumbered" in attrs:
            continue
        lvl = len(m.group(1))
        cnt[lvl - 1] += 1
        for j in range(lvl, 4):
            cnt[j] = 0
        num = ".".join(str(c) for c in cnt[:lvl])
        sid = re.search(r"#(sec:[^\s}]+)", attrs)
        if sid:
            ids[sid.group(1)] = ("Chapter " if lvl == 1 else "Section ") + num
    for i, t in enumerate(re.findall(r"\{#(fig:[^}\s]+)", md), 1):
        ids[t] = f"Figure {i}"
    for i, t in enumerate(re.findall(r"^: .*? \{#(tbl:[^}\s]+)\}", md, flags=re.M), 1):
        ids[t] = f"Table {i}"
    return ids


def restyle_tables(path):
    """Table cells get the compact 10 pt "Table Text" style (lists keep "Compact"); rows never split."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d = Document(path)
    for tb in d.tables:
        for row in tb.rows:
            pr = row._tr.get_or_add_trPr()          # keep each row on one page
            if pr.find(qn("w:cantSplit")) is None:
                pr.append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.style = d.styles["Table Text"]
        fit_columns(tb)
    d.save(path)


def fit_columns(tb, total_cm=15.5, cm_per_char=0.2):
    """Column widths from the content: every column first gets room for its longest word, then
    the remaining text width is shared in proportion to how much longer its cells are
    (pandoc otherwise gives equal widths)."""
    from docx.shared import Cm
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    ncol = len(tb.columns)
    longest, word = [1] * ncol, [1] * ncol
    for i, row in enumerate(tb.rows):
        for j, cell in enumerate(row.cells[:ncol]):
            t = cell.text.strip()
            if i > 0:                                   # header text does not claim extra width
                longest[j] = max(longest[j], min(len(t), 45))
            word[j] = max([word[j]] + [len(w) for w in t.split()])
    need = [w * cm_per_char + 0.45 for w in word]
    if sum(need) >= total_cm:
        width = [n * total_cm / sum(need) for n in need]
    else:
        flex = [max(L - w, 0) + 1 for L, w in zip(longest, word)]
        width = [n + (total_cm - sum(need)) * f / sum(flex) for n, f in zip(need, flex)]
    grid = tb._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(width[j] / 2.54 * 1440)))
    for row in tb.rows:
        for j, cell in enumerate(row.cells[:ncol]):
            cell.width = Cm(width[j])
    pr = tb._tbl.tblPr
    tw = pr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW")
        pr.append(tw)
    tw.set(qn("w:type"), "dxa")
    tw.set(qn("w:w"), str(int(sum(width) / 2.54 * 1440)))


def lists(md):
    """Lists of figures and tables, numbered in order of appearance (as pandoc-crossref does)."""
    figs = re.findall(r"!\[(.*?)\]\([^)]*\)\{#fig:[^}]*\}", md, flags=re.S)
    tbls = re.findall(r"^: (.*?) \{#tbl:[^}]*\}", md, flags=re.M)
    short = lambda c: re.sub(r"\s+", " ", re.split(r"(?<=[.])\s", c.strip())[0]).strip()
    block = lambda kind, caps: ('::: {custom-style="List Entry"}\n'
                                + "\n\n".join(f"{kind} {i}. {short(c)}" for i, c in enumerate(caps, 1))
                                + "\n:::")
    return block("Figure", figs), block("Table", tbls)


def fmt(v, spec):
    if spec == "date":
        return str(v)[:10]
    if spec is None:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if v == int(v) and abs(v) < 1e7:
                return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
            if abs(v) < 0.001:
                return f"{v:g}"
            return f"{v:.4f}".rstrip("0").rstrip(".")
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 10000 else str(v)
        return str(v)
    if spec == "int":
        return f"{int(round(v)):,}"
    if spec.startswith("pct"):
        return f"{100 * v:.{int(spec[3:] or 1)}f} %"
    if spec.startswith("sci"):
        s = f"{v:.{int(spec[3:] or 1)}e}"
        m, e = s.split("e")
        return f"{m} × 10^{int(e)}^"
    if spec == "signed3":
        return f"{v:+.3f}"
    return format(v, spec)


def substitute(text, ev, used):
    def rep(m):
        key, spec = m.group(1), m.group(2)
        if key not in ev:
            raise KeyError(key)
        used[key] = ev[key]
        v = ev[key]["value"]
        if v is None:
            raise ValueError(f"evidence {key} is empty (NaN/None)")
        return fmt(v, spec)
    return PH.sub(rep, text)


def main():
    ev = json.load(open(os.path.join(TH, "evidence", "evidence.json")))
    srcs = sorted(f for f in glob.glob(os.path.join(TH, "src", "*.md"))
                  if not f.endswith("99_evidence_appendix.md"))
    used, parts, missing = {}, [], []
    for f in srcs:
        txt = open(f, encoding="utf-8").read()
        try:
            parts.append(substitute(txt, ev, used))
        except (KeyError, ValueError) as e:
            # collect every missing key in this file before failing
            for m in PH.finditer(txt):
                if m.group(1) not in ev or ev[m.group(1)]["value"] is None:
                    missing.append((os.path.basename(f), m.group(1)))
    if missing:
        for f, k in missing:
            print(f"MISSING evidence: {f}: {k}")
        raise SystemExit(f"{len(missing)} placeholders without evidence; build aborted")
    # generated appendix: every number used, with its source
    rows = ["\\newpage", "", "# Appendix E: Evidence map for the numbers in this thesis {.unnumbered}", "",
            "Every number printed in this thesis was inserted from the evidence file "
            "`thesis/evidence/evidence.json` by the build script `thesis/tools/render.py`. "
            "That file is generated from the project's result files by "
            "`thesis/tools/build_evidence.py`. The table lists each value used, how it was "
            "obtained, and its source. *measured* means read from a result or log file; "
            "*derived* means computed from measured values by the stated rule; *config* means "
            "taken from code or configuration; *literature* means reported by a cited "
            "publication. Constants that *define* a method (equation coefficients, grid values, "
            "architecture sizes quoted with their module) are taken from the code at commit "
            "`2525367` and are not results; they are therefore not listed here.", "",
            "| Evidence key | Value | Kind | Source |", "|---|---|---|---|"]
    for k in sorted(used):
        e = used[k]
        v = e["value"]
        vs = f"{v:.6g}" if isinstance(v, float) else str(v)
        src = e["source"].replace("|", "/")
        note = f" ({e['note']})" if e.get("note") else ""
        rows.append(f"| `{k}` | {vs[:60]} | {e['kind']}{note} | `{src}` |")
    app = "\n".join(rows) + "\n"
    with open(os.path.join(TH, "src", "99_evidence_appendix.md"), "w") as f:
        f.write("<!-- generated by thesis/tools/render.py; do not edit -->\n" + app)
    os.makedirs(os.path.join(TH, "build"), exist_ok=True)
    md = os.path.join(TH, "build", "COSMOS_Thesis.md")
    body = "\n\n".join(parts) + "\n\n" + app
    out_lines, prev = [], ""
    for line in body.splitlines():
        if line.strip() == "\\newpage":
            if prev != "<<PB>>":
                out_lines.append("<<PB>>")
                prev = "<<PB>>"
            continue
        if line.startswith("# ") and prev != "<<PB>>" and out_lines:
            out_lines += ["<<PB>>", ""]
        out_lines.append(line)
        if line.strip():
            prev = line if line != "<<PB>>" else prev
    body = "\n".join(out_lines).replace("<<PB>>", PAGEBREAK)
    lof, lot = lists(body)
    body = body.replace("<!-- LOF -->", lof).replace("<!-- LOT -->", lot)
    body = body.replace("<!-- TOC -->", TOCFIELD)
    with open(md, "w", encoding="utf-8") as f:
        f.write(body)
    json.dump(used, open(os.path.join(TH, "build", "used_keys.json"), "w"), indent=1)
    out = os.path.join(TH, "build", "COSMOS_Thesis.docx")
    cmd = [PANDOC, md, "--from", "markdown+tex_math_dollars+pipe_tables+grid_tables",
           "--filter", XREF, "--citeproc",
           "--bibliography", os.path.join(TH, "refs.bib"),
           "--csl", os.path.join(HERE, "ieee.csl"),
           "--reference-doc", os.path.join(HERE, "reference.docx"),
           "--resource-path", TH, "--number-sections", "--columns=10000",
           "-M", "figureTitle=Figure", "-M", "tableTitle=Table", "-M", "eqnPrefix=Eq.",
           "-M", "figPrefix=Figure", "-M", "tblPrefix=Table", "-M", "secPrefix=Section",
           "-M", "link-citations=true", "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=TH)
    if r.returncode:
        print(r.stderr)
        raise SystemExit("pandoc failed")
    restyle_tables(out)
    warn = [l for l in r.stderr.splitlines() if l.strip()]
    for l in warn[:40]:
        print("pandoc:", l)
    # stand-alone deliverable: Appendix D (contribution, gaps, readiness, roadmap)
    m = re.search(r"(# Appendix D:.*?)(?=\n```\{=openxml\}|\n# Appendix E)", body, flags=re.S)
    if m:
        ids = numbering(body)
        part = m.group(1)
        part = re.sub(r"\[@((?:sec|tbl|fig|eq):[^\]\s;]+)\]",
                      lambda mm: f"thesis {ids.get(mm.group(1), mm.group(1))}", part)
        q1md = os.path.join(TH, "build", "Q1_Readiness_and_Roadmap.md")
        with open(q1md, "w", encoding="utf-8") as f:
            f.write("---\ntitle: \"COSMOS thesis: contribution, evidence gaps, Q1 readiness and roadmap\"\n---\n\n"
                    + part.replace("{.unnumbered}", "").replace("{#sec:appendix-q1 }", "") + "\n\n# References\n")
        q1 = os.path.join(TH, "build", "Q1_Readiness_and_Roadmap.docx")
        r2 = subprocess.run([PANDOC, q1md, "--filter", XREF, "--citeproc", "--bibliography",
                             os.path.join(TH, "refs.bib"), "--csl", os.path.join(HERE, "ieee.csl"),
                             "--reference-doc", os.path.join(HERE, "reference.docx"), "-o", q1],
                            capture_output=True, text=True, cwd=TH)
        print("wrote", q1, "" if not r2.returncode else r2.stderr[:300])
    print(f"{len(used)} evidence values used; wrote {out}")


if __name__ == "__main__":
    main()
